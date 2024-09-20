import os
import re
import threading
import torch
import torch.nn as nn
from typing import Tuple, Deque
from collections import deque
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from ..utils import (
    BLOCK_TEMPLATE,
    ATTN_SAVE_PATH,
    MLP_SAVE_PATH,
    INPUT_SAVE_PATH,
    OUTPUT_SAVE_PATH,
)

torch.set_num_threads(8)


class MemoryManager:
    """
    Manage the loading and unloading of model weights from disk to cpu memory.
    """

    def __init__(self, model, rank, args):
        self._model = model
        self._device = args.device
        self._rank = rank
        self._split_dir = os.path.join(args.model_path, args.save_dir, f"node_{rank}")
        _all_blocks = [
            BLOCK_TEMPLATE.format(l=block_idx, type=block_type)
            for block_idx in range(model.config.num_hidden_layers)
            for block_type in ["self_attn", "mlp"]
        ]
        self._all_blocks = ["input"] + _all_blocks + ["output"] if rank == 0 else _all_blocks
        self._loaded_blocks: Deque[str] = deque(maxlen=args.memory_window)  # tracks loaded blocks
        self._layers_in_block = {block_key: [] for block_key in self._all_blocks}
        self._disabled = args.disable_memory_schedule  # whether to disable memory schedule
        self._batch_loaded = False
        self._futures: Deque[Future] = {}
        self._condition = threading.Condition()
        self._stop = False  # flag to stop the scheduling thread
        self._thread = None

    def _get_bid_and_btype(self, block_name: str) -> Tuple[int, str]:
        """
        Extract the block id and block type from the given block name.

        Args:
            block_name (str): The name of the block, following the BLOCK_TEMPLATE pattern.

        Returns:
            Tuple[int, str]: A tuple containing the block id (int) and block type (str).
        """
        pattern = BLOCK_TEMPLATE.format(type=r'(\w+)', l=r'(\d+)')
        match = re.match(pattern, block_name)
        if match:
            return int(match.group(2)), match.group(1)
        else:
            raise ValueError(f"Key '{block_name}' does not match pattern '{pattern}'")

    def _find_module(self, model, key: str) -> nn.Module:
        """
        Find the module corresponding to the given key.

        Args:
            model: The model containing the module.
            key (str): The full key path to the module (e.g., 'model.embed_tokens.weight').

        Returns:
            The module object if found.
       """
        module = model
        *module_names, param_name = key.split('.')
        for name in module_names:
            module = getattr(module, name, None)
            if module is None:
                raise ValueError(f"Parameter {key} not found.")
        return module

    def _load_key(self, block_name: str, key: str, weight: torch.Tensor):
        """
        Load the weights into the model for a given key.

        Args:
            block_name (str): The name of the block, can be "input", "self_attn", "mlp", or "output".
            key (str): The key specifying where the weight should be loaded.
            weight: The weight tensor to load.
        """
        *module_names, param_name = key.split('.')
        try:
            module = self._find_module(self._model, key)
        except ValueError:  # key not exist in model parameters
            return
        # register the parameter, note that only float32 is supported on cpu
        module.register_parameter(
            param_name,
            torch.nn.Parameter(weight.float(), requires_grad=False)
        )
        # record which layers are in a block
        if key not in self._layers_in_block[block_name]:
            self._layers_in_block[block_name].append(key)

    def _load_block(self, block_name: str):
        """
        Load the weights for a specific block.

        Args:
            block_name (str): The name of the block to load.
        """
        # determine the path to the binary file
        if block_name == "input":
            bin_path = os.path.join(self._split_dir, INPUT_SAVE_PATH)
        elif block_name == "output":
            bin_path = os.path.join(self._split_dir, OUTPUT_SAVE_PATH)
        elif "self_attn" in block_name:
            block_id, _ = self._get_bid_and_btype(block_name)
            bin_path = os.path.join(self._split_dir, ATTN_SAVE_PATH.format(l=block_id))
        elif "mlp" in block_name:
            block_id, _ = self._get_bid_and_btype(block_name)
            bin_path = os.path.join(self._split_dir, MLP_SAVE_PATH.format(l=block_id))
        else:
            raise NotImplementedError(f"Block name {block_name} is not supported.")

        try:
            # load pretrained weights into memory
            with torch.no_grad(), open(bin_path, 'rb') as f:
                pretrained_weights = torch.load(f, map_location=self._device)

            # use a thread pool to load weights concurrently
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(self._load_key, block_name, k, v): k
                           for k, v in pretrained_weights.items()}
                # wait for all nested futures to complete
                for future in as_completed(futures):
                    future.result()

            del pretrained_weights
        except FileNotFoundError:
            if block_name not in ("input", "output"):
                raise FileNotFoundError(f"Binary file {bin_path} not found.")

    def _daemon_loop(self):
        """
        The daemon loop function for loading model weights.
        """
        # load blocks
        block_idx = 0
        while True:
            # load the next block
            block_name = self._all_blocks[block_idx]
            block_idx = (block_idx + 1) % len(self._all_blocks)

            # wait until there is space available
            with self._condition:
                while not self._disabled and len(self._loaded_blocks) >= self._loaded_blocks.maxlen:
                    self._condition.wait()

            if self._stop:  # exit loop to exit the memory scheduling thread
                return

            # load the blocks asynchronously
            with ThreadPoolExecutor() as executor:
                future = executor.submit(self._load_block, block_name)

            # add the future to the tracking list
            with self._condition:
                self._futures[block_name] = future
                self._loaded_blocks.append(block_name)
                self._condition.notify_all()

            # exist the thread at the end of the first epoch if memory schedule is disabled
            if self._disabled and block_idx == 0:
                self._batch_loaded = True
                return

    def _release_block(self, block_name: str):
        """
        Delete used weights in a given block to save memory.

        Args:
            block_name (str): The name of the block to release.
        """
        if block_name not in self._layers_in_block:
            raise KeyError(f"Block name '{block_name}' not found in _layers_in_block.")

        # a block may contain multiple keys, delete them
        for layer_key in self._layers_in_block[block_name]:
            module = self._find_module(self._model, layer_key)
            *module_names, param_name = layer_key.split('.')
            if module._parameters[param_name].device.type == "cuda":
                # clear data in cuda memory, todo: this feature is not tested
                with torch.no_grad():
                    module._parameters[param_name].data = None
                torch.cuda.empty_cache()
            else:
                # clear data in cpu memory
                del module._parameters[param_name]

    def release_before(self, block_name: str):
        """
        Release all used blocks before this block.

        Args:
            block_name (str): The name of the block before which used blocks are deleted.
        """
        if self._disabled:
            return

        with self._condition:
            while len(self._loaded_blocks) > 0:
                current_block = self._loaded_blocks[0]
                if block_name == self._all_blocks[0]:
                    # clear all loaded blocks except the current one
                    if current_block == block_name:
                        break
                    else:
                        self._release_block(self._loaded_blocks.popleft())
                else:
                    # clear used blocks before this one
                    if self._all_blocks.index(current_block) < self._all_blocks.index(block_name):
                        self._release_block(self._loaded_blocks.popleft())
                    else:
                        break
            self._condition.notify_all()

    def start(self) -> Future:
        """
        Start the memory scheduler in the background.

        Returns:
            Future: The future object representing the memory scheduler thread.
        """
        if self._thread is None:
            executor = ThreadPoolExecutor()
            self._thread = executor.submit(self._daemon_loop)

    def wait(self, block_name: str):
        """
        Wait for the block to complete.

        Args:
            block_name (str): The name of the block to wait for.
        """
        if not self._batch_loaded:
            with self._condition:
                while block_name not in self._futures:
                    self._condition.wait()
                while not self._futures[block_name].done():
                    self._condition.wait()

            with self._condition:
                del self._futures[block_name]
                self._condition.notify_all()

    @contextmanager
    def wait_and_release(self, block_name: str):
        """
        Context manager that waits for a block to load, executes the context, and releases the block.

        Args:
            block_name (str): The name of the block to wait and release.
        """
        self.wait(block_name)
        try:
            yield  # execution of statements happens here
        finally:
            self.release_before(block_name)

    def stop(self):
        """
        Stop the memory scheduler thread in a graceful manner.
        """
        with self._condition:
            self._stop = True
            self._loaded_blocks.clear()
            self._condition.notify_all()
