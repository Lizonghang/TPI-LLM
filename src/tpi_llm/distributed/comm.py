import socket
import pickle
import struct
import logging
import torch
import mxnet as mx
from mxnet import nd
from mxnet.kvstore import KVStore
from abc import ABC, abstractmethod
from .utils import connect_with_retry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class CommunicatorBase(ABC):
    """
    Base class for implementing communication between distributed nodes.
    This class provides the basic functionality of allreduce and acts as an abstract base
    for more specific communicator implementations, such as broadcast, request, and barrier.

    Args:
        kvstore (KVStore): KVStore used for distributed key-value storage and synchronization.
    """

    def __init__(self, kvstore: KVStore):
        self._kv = kvstore
        self._s = None

    def allreduce(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Perform an allreduce operation using KVStore on a PyTorch tensor across distributed nodes.

        Args:
            tensor (torch.Tensor): The tensor to be all-reduced across nodes.

        Returns:
            torch.Tensor: The reduced tensor, synchronized across all nodes.
        """
        # convert torch.Tensor to a DLPack tensor and then to an MXNet NDArray
        tensor_np = tensor.detach().cpu().numpy()
        tensor_nd = nd.array(tensor_np, ctx=mx.cpu())
        # perform allreduce, 1 for prefilling and 0 for decoding
        key = str(int(tensor.size(1) > 1))
        self._kv.push(key, tensor_nd)
        self._kv.pull(key, out=tensor_nd)
        # convert the reduced MXNet NDArray back to a PyTorch tensor
        return torch.from_numpy(tensor_nd.asnumpy()).to(tensor.device)

    @abstractmethod
    def barrier(self):
        """
        Abstract method for synchronizing nodes. Must be implemented by derived classes.
        """
        pass

    def close(self):
        """
        Clean up the socket connection.
        """
        if self._s is not None:
            self._s.close()


class CommunicatorMaster(CommunicatorBase):
    """
    Communicator implementation for the master node. Handles data broadcasting and synchronization
    among distributed nodes.

    Args:
        kvstore (KVStore): KVStore used for distributed key-value storage and synchronization.
        host (str): The host IP address for the master node.
        port (int): The port on which the master node listens for client connections.
        world_size (int): The total number of nodes in the distributed system.
    """
    def __init__(self, kvstore: KVStore, host: str, port: int, world_size: int):
        super().__init__(kvstore)
        self._world_size = world_size
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._s.bind((host, port))
        self._s.listen(world_size - 1)  # listen for all other nodes
        self._client_sockets = {}

    def _collect_sockets(self):
        while len(self._client_sockets) < self._world_size - 1:
            s, _ = self._s.accept()
            rank = s.recv(4)
            rank = struct.unpack("i", rank)[0]
            self._client_sockets[s] = rank
            logger.info(f"Rank {rank} connected to me.")

    def broadcast(self, data):
        """
        Broadcast data to all other nodes.

        Args:
            data: The data to be broadcast.
        """
        # collect client connections first
        self._collect_sockets()

        # send the data to all connected clients
        if isinstance(data, int):
            serialized_data = struct.pack("i", data)
        else:
            serialized_data = pickle.dumps(data)
        data_len = struct.pack("i", len(serialized_data))
        for s in self._client_sockets.keys():
            s.sendall(data_len)  # meta head
            s.sendall(serialized_data)  # data

    def barrier(self):
        """
        Synchronize all nodes by implementing a barrier. Collects BARRIER requests from workers
        and releases them once all have reached the barrier.
        """
        # collect client connections first
        self._collect_sockets()

        # collect BARRIER messages from all clients
        barrier_clients = set(self._client_sockets.keys())
        received_clients = set()

        while received_clients != barrier_clients:
            for s in barrier_clients:
                msg = s.recv(7).decode("utf-8")
                assert msg == "BARRIER", f"Received an unexpected message {msg}."
                assert s not in received_clients, \
                    (f"Socket of rank {self._client_sockets[s]} has already been "
                     f"barried! It may be barried twice.")
                received_clients.add(s)

        # send ACK to each client to release them from the barrier
        for s in received_clients:
            s.sendall("ACK".encode("utf-8"))


class CommunicatorClient(CommunicatorBase):
    """
    Communicator implementation for non-master nodes. Handles requests and synchronization
    operations with the master node.

    Args:
        kvstore (KVStore): KVStore used for distributed key-value storage and synchronization.
        host (str): The host IP address of the master node.
        port (int): The port on which the master node is listening.
        rank (int): My rank.
    """
    def __init__(self, kvstore: KVStore, host: str, port: int, rank: int):
        super().__init__(kvstore)
        self._host = host
        self._port = port
        self._rank = rank

    def request(self):
        """
        Request and receive data from the master node.

        Returns:
            The broadcast data received from the master node.
        """
        # establish a long-term connection, if not already connected
        if self._s is None:
            self._s = connect_with_retry(self._host, self._port, self._rank)

        # receive data length from meta info
        data_len = self._s.recv(4)
        data_len = struct.unpack("i", data_len)[0]

        # receive the data in chunks from the master node
        data = b""
        while len(data) < data_len:
            packet = self._s.recv(min(4096, data_len - len(data)))
            if not packet:
                break
            data += packet

        if data_len == 4:  # an integer received
            return struct.unpack("i", data)[0]
        return pickle.loads(data)  # a pickle object received

    def barrier(self):
        """
        Synchronize with the master node by sending a BARRIER request and waiting for an ACK.
        """
        # establish a long-term connection, if not already connected
        if self._s is None:
            self._s = connect_with_retry(self._host, self._port, self._rank)

        # send BARRIER message
        self._s.sendall("BARRIER".encode("utf-8"))

        # wait for ACK response to exit the barrier
        ack = self._s.recv(3).decode("utf-8")
        assert ack == "ACK", f"Received an unexpected message {ack}."
