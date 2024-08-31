import socket
import pickle
import torch
import mxnet as mx
from mxnet import nd
from mxnet.kvstore import KVStore


def server_broadcast(broadcast_data, master_ip, broadcast_port, world_size, kvstore):
    """
    The master node broadcasts data to all other nodes.

    Args:
        broadcast_data (any): The data to broadcast.
        master_ip (str): The IP address of the master node.
        broadcast_port (int): The port on which the master node will listen for connections.
        world_size (int): The total number of nodes.
        kvstore (KVStore): A key-value store object for managing synchronization between nodes.
    """
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # set SO_REUSEADDR to release the socket immediately after the socket is closed
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((master_ip, broadcast_port))
    server_socket.listen(world_size - 1)  # listen for all other nodes
    kvstore._barrier()  # make sure the master node starts listening before other nodes connect

    def _handle_client(client_socket):
        client_socket.sendall(pickle.dumps(broadcast_data))
        client_socket.close()

    connected_clients = 0
    while connected_clients < world_size - 1:
        client_socket, _ = server_socket.accept()
        _handle_client(client_socket)
        connected_clients += 1
    server_socket.close()


def client_request(master_ip, broadcast_port, kvstore):
    """
    Non-master nodes request and receive data from the master node.

    Args:
        master_ip (str): The IP address of the master node.
        broadcast_port (int): The port on which the master node is listening for connections.
        kvstore (KVStore): A key-value store object for managing synchronization between nodes.

    Returns:
        broadcast_data: The broadcast data received from the master node.
    """
    kvstore._barrier()  # make sure the master node starts listening before other nodes connect
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((master_ip, broadcast_port))

    # receive the data in chunks from the master node
    data = b""
    while True:
        packet = client_socket.recv(4096)
        if not packet:
            break
        data += packet
    client_socket.close()

    broadcast_data = pickle.loads(data)
    return broadcast_data


def allreduce(kvstore: KVStore, tensor: torch.Tensor) -> torch.Tensor:
    """
    Perform an allreduce operation using KVStore on a PyTorch tensor across distributed nodes.

    Args:
        kvstore (KVStore): The key-value store object for distributed communication.
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
        kvstore.push(key, tensor_nd)
        kvstore.pull(key, out=tensor_nd)
    except mx.base.MXNetError:  # the key may not be initialized
        kvstore.init(key, nd.zeros_like(tensor_nd))  # only kv.rank 0 will execute initialization
        kvstore._barrier()  # make sure kvstore init is complete before retrying
        kvstore.push(key, tensor_nd)
        kvstore.pull(key, out=tensor_nd)

    # convert the reduced MXNet NDArray back to a PyTorch tensor
    return torch.from_numpy(tensor_nd.asnumpy()).to(tensor.device)
