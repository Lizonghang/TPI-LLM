import torch
import torch.distributed as dist
from typing import List, Union
from mxnet import nd
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
        async_op: bool = False,
        use_kvstore: bool = False
    ) -> Union[dist.distributed_c10d.Work, None]:
        """
        Reduce the tensor data across all nodes such that all nodes obtain the reduced tensor.
        By default, it performs a sum reduction.

        Args:
            tensor (torch.Tensor): The tensor to be reduced.
            op (ReduceOp, optional): The reduction operation to be applied. Defaults to ReduceOp.SUM.
            async_op (bool, optional): If True, the operation will be performed asynchronously, and a
                `dist.distributed_c10d.Work` object will be returned (default is False).
            use_kvstore (bool, optional): If True, the tensor will be handled using MXNET / NetStorm
                KVStore.

        Returns:
            Union[dist.distributed_c10d.Work, None]: Returns a `Work` object if `async_op` is True;
            otherwise, returns None.
        """
        # use torch.distributed to perform allreduce
        if not use_kvstore:
            return dist.all_reduce(tensor, op=op, async_op=async_op)

        # use kvstore to perform allreduce
        # todo: convert torch.Tensor to nd.NDArray and convert it back after allreduce
        pass

        # for idx, param in enumerate(params):  # idex=模型的层号
        #     if param.grad_req == "null":
        #         continue
        #     kvstore_dist.push(idx, param.grad() / num_samples, priority=-idx)
        #
        # for idx, param in enumerate(params):
        #     if param.grad_req == "null":
        #         continue
        #     temp = nd.zeros(param.shape, ctx=ctx)
        #     kvstore_dist.pull(idx, temp, priority=-idx)
        #     temp.wait_to_read()
        #     param.grad()[:] = temp
        # nd.waitall()
