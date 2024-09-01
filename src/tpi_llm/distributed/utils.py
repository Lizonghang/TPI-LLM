import time


def connect_with_retry(s, host, port, retry_interval=0.5, max_retry=10):
    """
    This function tries to establish a connection to the given host and port using
    the provided socket object. If the connection fails due to a `ConnectionRefusedError`,
    it will retry the connection after a specified interval, up to a maximum number of retries.

    Args:
        s (socket.socket): The socket object to be used for the connection.
        host (str): The hostname or IP address of the node to connect to.
        port (int): The port number on the node to connect to.
        retry_interval (float, optional): The time in seconds to wait between retry attempts.
                                          Defaults to 0.5 seconds.
        max_retry (int, optional): The maximum number of retry attempts before giving up.
                                   Defaults to 10 attempts.

    Raises:
        ConnectionRefusedError: If the connection is refused after the maximum number of retries.
    """
    num_retry = 0
    while num_retry < max_retry:
        num_retry += 1
        try:
            s.connect((host, port))
        except ConnectionRefusedError:
            time.sleep(retry_interval)
