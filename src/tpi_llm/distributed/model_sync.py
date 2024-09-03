import os
import struct
import logging
import socket
import threading
from typing import List
from tqdm import tqdm
from ..utils import FILES_TO_SYNC

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def download_file(host: str, port: int, rank: int, model_path: str, split_path: str):
    """
    Connect to the file server on the master node to download sliced model parameter files.

    Args:
        host (str): The IP address or hostname of the master node.
        port (int): The port number on which the file server is listening.
        rank (int): The rank of the requesting node.
        model_path (str): The directory to save the main model files.
        split_path (str): The directory to save the split model files.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            # connect to the file server
            s.connect((host, port))
            # send my rank to the master node
            s.sendall(struct.pack("i", rank))

            # receive the number of files to be received
            num_files_data = s.recv(4)
            num_files = struct.unpack("i", num_files_data)[0]
            logger.info(f"Downloading {num_files} files from the master node.")

            # download model files
            for _ in tqdm(range(num_files), leave=True):
                # receive file name length
                file_name_length_data = s.recv(4)
                file_name_length = struct.unpack('i', file_name_length_data)[0]
                # receive file name
                file_name = s.recv(file_name_length).decode("utf-8")
                # receive file size
                file_size_data = s.recv(8)
                file_size = struct.unpack("Q", file_size_data)[0]
                # receive file content
                save_path_ = os.path.join(model_path, split_path, f"node_{rank}") \
                    if ".bin" in file_name else model_path
                with open(os.path.join(save_path_, file_name), "wb") as f:
                    bytes_received = 0
                    while bytes_received < file_size:
                        data = s.recv(min(1024, file_size - bytes_received))
                        if not data:
                            break
                        f.write(data)
                        bytes_received += len(data)
        except ConnectionRefusedError:
            raise ConnectionRefusedError(f"Rank {rank} connected to {host}:{port} failed.")


def run_sync_server(host: str, port: int, model_path: str, split_path: str):
    """
    Start a file server that sends sliced model parameter files to requesting nodes.

    Args:
        host (str): The IP address or hostname to bind the file server to.
        port (int): The port number to bind the file server to.
        model_path (str): The directory containing the main model files.
        split_path (str): The directory containing the split model files.
    """
    def _get_files_to_send(rank: int):
        """
        Retrieve the list of files to be sent to node <rank>.

        Args:
            rank (int): The rank of the requesting node.

        Returns:
            List[str]: A list of file paths to be sent.
        """
        files_in_dir = set(os.listdir(model_path))
        files_to_send = files_in_dir.intersection(FILES_TO_SYNC)
        node_file_path = os.path.join(split_path, f"node_{rank}")
        split_weights_in_dir = os.listdir(node_file_path)
        return ([os.path.join(model_path, fn) for fn in files_to_send]
                + [os.path.join(node_file_path, fn) for fn in split_weights_in_dir])

    def _send_files(conn, files: List[str]):
        """
        Send selected files to a connected node.

        Args:
            conn (socket.socket): The connection socket to the node.
            files (List[str]): A list of files to be sent.
        """
        # send the number of files
        conn.sendall(struct.pack("i", len(files)))
        # send each file name and content
        for file_path in files:
            # send file name length and file name
            file_name = os.path.basename(file_path)
            conn.sendall(struct.pack("i", len(file_name)))
            conn.sendall(file_name.encode('utf-8'))
            # send file size
            file_size = os.path.getsize(file_path)
            conn.sendall(struct.pack("Q", file_size))
            # send file content
            with open(file_path, "rb") as f:
                bytes_read = 0
                while bytes_read < file_size:
                    data = f.read(min(1024, file_size - bytes_read))
                    if not data:
                        break
                    conn.sendall(data)
                    bytes_read += len(data)

    def _download_handler(conn):
        """
        Handle a download request from other nodes.

        Args:
            conn (socket.socket): The connection socket to the client.
        """
        # receive the rank from the requesting node
        rank_data = conn.recv(4)
        rank = struct.unpack("i", rank_data)[0]
        logger.info(f"Received file request from node {rank}.")
        # get files to be sent
        files_to_send = _get_files_to_send(rank)
        # send files to the requesting node
        _send_files(conn, files_to_send)

    def _download_listener():
        """
        Listen for incoming connections and file downloading requests.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            s.listen(20)  # Listen for up to 20 connections
            while True:
                conn, addr = s.accept()
                client_thread = threading.Thread(target=_download_handler, args=(conn,))
                client_thread.start()

    server_thread = threading.Thread(target=_download_listener)
    server_thread.daemon = True
    server_thread.start()
