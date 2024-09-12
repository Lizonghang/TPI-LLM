import socket
import pickle
import torch
import mxnet as mx
from mxnet import nd
from mxnet.kvstore import KVStore
from abc import ABC, abstractmethod
from .utils import connect_with_retry


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

    @abstractmethod
    def close(self):
        """
        Abstract method for cleaning up the socket connection. Must be implemented by derived classes.
        """
        pass


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

    def broadcast(self, data):
        """
        Broadcast data to all other nodes.

        Args:
            data: The data to be broadcast.
        """
        # collect client connections first
        client_sockets = []
        while len(client_sockets) < self._world_size - 1:
            client_socket, _ = self._s.accept()
            client_sockets.append(client_socket)

        # send the data to all connected clients
        serialized_data = pickle.dumps(data)
        for client_socket in client_sockets:
            client_socket.sendall(serialized_data)
            client_socket.close()

    def barrier(self):
        """
        Synchronize all nodes by implementing a barrier. Collects BARRIER requests from workers
        and releases them once all have reached the barrier.
        """
        # collect BARRIER messages from all clients
        barrier_clients = []
        while len(barrier_clients) < self._world_size - 1:
            client_socket, _ = self._s.accept()
            message = client_socket.recv(8).decode("utf-8")
            if message == "BARRIER":
                barrier_clients.append(client_socket)
            else:
                raise ValueError(f"Received an unexpected message {message}.")

        # send ACK to each client to release them from the barrier
        for client in barrier_clients:
            client.sendall("ACK".encode("utf-8"))
            client.close()

    def close(self):
        self._s.close()


class CommunicatorClient(CommunicatorBase):
    """
    Communicator implementation for non-master nodes. Handles requests and synchronization
    operations with the master node.

    Args:
        kvstore (KVStore): KVStore used for distributed key-value storage and synchronization.
        host (str): The host IP address of the master node.
        port (int): The port on which the master node is listening.
    """
    def __init__(self, kvstore: KVStore, host: str, port: int):
        super().__init__(kvstore)
        self._host = host
        self._port = port

    def request(self):
        """
        Request and receive data from the master node.

        Returns:
            The broadcast data received from the master node.
        """
        s = connect_with_retry(self._host, self._port)
        # receive the data in chunks from the master node
        data = b""
        while True:
            packet = s.recv(4096)
            if not packet:
                break
            data += packet
        s.close()
        return pickle.loads(data)

    def barrier(self):
        """
        Synchronize with the master node by sending a BARRIER request and waiting for an ACK.
        """
        # connect to the master and send BARRIER message
        s = connect_with_retry(self._host, self._port)
        s.sendall("BARRIER".encode("utf-8"))

        # wait for ACK response to exit the barrier
        ack = s.recv(8).decode("utf-8")
        if ack == "ACK":
            s.close()
        else:
            raise ValueError(f"Received an unexpected message {ack}.")

    def close(self):
        # nothing to do with client
        pass
