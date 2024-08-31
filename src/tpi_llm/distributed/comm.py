import socket
import pickle
import torch
import mxnet as mx
from mxnet import nd
from mxnet.kvstore import KVStore
from abc import ABC, abstractmethod


class CommunicatorBase(ABC):

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
        try:
            self._kv.push(key, tensor_nd)
            self._kv.pull(key, out=tensor_nd)
        except mx.base.MXNetError:  # the key may not be initialized
            self._kv.init(key, nd.zeros_like(tensor_nd))  # only kv.rank 0 will execute initialization
            self.barrier()  # make sure kvstore init is complete before retrying
            self._kv.push(key, tensor_nd)
            self._kv.pull(key, out=tensor_nd)

        # convert the reduced MXNet NDArray back to a PyTorch tensor
        return torch.from_numpy(tensor_nd.asnumpy()).to(tensor.device)

    @abstractmethod
    def barrier(self):
        pass


class CommunicatorMaster(CommunicatorBase):

    def __init__(self, kvstore: KVStore, host: str, port: int, world_size: int):
        super().__init__(kvstore)
        self._world_size = world_size
        self._s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._s.bind((host, port))
        self._s.listen(5)  # listen for all other nodes

    def broadcast(self, data):
        connected_clients = 0
        while connected_clients < self._world_size - 1:
            client_socket, _ = self._s.accept()
            client_socket.sendall(pickle.dumps(data))
            client_socket.close()
            connected_clients += 1

    def barrier(self):
        self._kv._barrier()
        # # collect BARRIER messages from all clients
        # barrier_clients = []
        # while len(barrier_clients) < self._world_size - 1:
        #     client_socket, _ = self._s.accept()
        #     message = client_socket.recv(8).decode("utf-8")
        #     if message == "BARRIER":
        #         barrier_clients.append(client_socket)
        #     else:
        #         raise ValueError(f"Received an unexpected message {message}.")
        #
        # # send ACK to each client to release them from the barrier
        # for client in barrier_clients:
        #     client.sendall("ACK".encode("utf-8"))
        #     client.close()
        # barrier_clients.clear()


class CommunicatorClient(CommunicatorBase):

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
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((self._host, self._port))
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
        self._kv._barrier()
        # # connect to the master and send BARRIER message
        # s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # s.connect((self._host, self._port))
        # s.sendall("BARRIER".encode("utf-8"))
        #
        # # wait for ACK response to exit the barrier
        # ack = s.recv(8).decode("utf-8")
        # if ack == "ACK":
        #     s.close()
        # else:
        #     raise ValueError(f"Received an unexpected message {ack}.")
