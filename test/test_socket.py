import os
import time
import socket
import argparse
import threading


def _download_handler(addr):
    print("Accepted file download request from address", addr)
    time.sleep(60)


def _download_listener(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen(20)  # Listen for up to 20 connections
        while True:
            conn, addr = s.accept()
            client_thread = threading.Thread(target=_download_handler, args=(addr,))
            client_thread.start()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_port", type=int, default=29600, help="File server port.")
    parser.add_argument("--broadcast_port", type=int, default=29700, help="Broadcast server port.")
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world_size = 5
    master_ip = os.environ["MASTER_ADDR"]

    if rank == 0:
        print("In communicator ...")
        s1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s1.bind((master_ip, args.broadcast_port))
        s1.listen(world_size - 1)

        print("In file server ...")
        server_thread = threading.Thread(target=_download_listener, args=(master_ip, args.file_port))
        server_thread.daemon = True
        server_thread.start()

        print("In barrier ...")
    else:
        print("In barrier ...")
        print("Client is trying to connect the master ...")
        s2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s2.connect((master_ip, args.broadcast_port))
        print("Client connected to master success.")

    time.sleep(20)
