import torch
import torch.distributed as dist
from typing import List, Union
from torch._C._distributed_c10d import ReduceOp


class DistributedCommPrimitive:

    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size

    # todo(lizh): optimize the communication efficiency
    @classmethod
    def broadcast(cls, object_list: List[object], src: int):
        """
        Broadcast a list of objects from the source node to all other nodes.

        Args:
            object_list (List[object]): A list of objects to be broadcasted from the source node.
            src (int): The rank of the source node.
        """
        dist.broadcast_object_list(object_list, src=src)

    # todo(lizh): optimize the communication efficiency
    @classmethod
    def allreduce(
        cls,
        tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        async_op=False
    ) -> Union[dist.distributed_c10d.Work, None]:
        """
        Reduce the tensor data across all nodes such that all nodes obtain the reduced tensor.
        By default, it performs a sum reduction.

        Args:
            tensor (torch.Tensor): The tensor to be reduced.
            op (ReduceOp, optional): The reduction operation to be applied. Defaults to ReduceOp.SUM.
            async_op (bool, optional): If True, the operation will be performed asynchronously, and a
            `dist.distributed_c10d.Work` object will be returned (default is False).

        Returns:
            Union[dist.distributed_c10d.Work, None]: Returns a `Work` object if `async_op` is True;
            otherwise, returns None.
        """
        return dist.all_reduce(tensor, op=op, async_op=async_op)
