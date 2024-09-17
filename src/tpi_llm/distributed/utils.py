import socket
import time
import struct
import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def connect_with_retry(host, port, rank, retry_interval=0.5, max_retry=10) -> socket.socket:
    """
    This function tries to establish a connection to the given host and port using
    the provided socket object. If the connection fails due to a `ConnectionRefusedError`,
    it will retry the connection after a specified interval, up to a maximum number of retries.

    Args:
        host (str): The hostname or IP address of the node to connect to.
        port (int): The port number on the node to connect to.
        rank (int): My rank.
        retry_interval (float, optional): The time in seconds to wait between retry attempts.
                                          Defaults to 0.5 seconds.
        max_retry (int, optional): The maximum number of retry attempts before giving up.
                                   Defaults to 10 attempts.

    Raises:
        ConnectionRefusedError: If the connection is refused after the maximum number of retries.

    Returns:
        socket.socket: The connected socket object.
    """
    num_retry = 0
    while num_retry < max_retry:
        num_retry += 1
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.connect((host, port))
            s.sendall(struct.pack("i", rank))
            logger.info(f"Connected to the master node ({host}:{port}).")
            return s
        except ConnectionRefusedError:
            s.close()
            time.sleep(retry_interval)

    raise ConnectionRefusedError(f"Failed to connect to {host}:{port} after {max_retry} attempts.")


def recv_data_chunk(conn, data_len):
    """
    Receives a chunk of data from a socket connection until the specified length is reached.

    Parameters:
        conn (socket object): The socket connection from which to receive data.
        data_len (int): The expected length of the data to receive.

    Returns:
        bytes: The received data.
    """
    data = b""
    while len(data) < data_len:
        packet = conn.recv(min(4096, data_len - len(data)))
        if not packet:
            break
        data += packet
    return data
