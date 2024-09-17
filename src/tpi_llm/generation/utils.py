import inspect
import torch
from torch import nn
from typing import Optional, List, Callable, Union
from transformers import (
    GenerationMixin,
    GenerationConfig,
    LogitsProcessorList,
    StoppingCriteriaList,
    DynamicCache,
)
from ..distributed import CommunicatorBase


class TPIGenerationMixin(GenerationMixin):
    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        streamer: Optional["BaseStreamer"],
        communicator: Optional[Union[CommunicatorBase, "module"]],
        logits_warper: Optional[LogitsProcessorList],
        **model_kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token IDs for models with a language modeling head using multinomial sampling
        or greedy decoding. This function is designed to handle distributed generation, where the master node
        (rank 0) leads the generation process, and non-master nodes (other ranks) perform tensor parallelism
        calculation.

        Parameters:
            input_ids (torch.LongTensor):
                The sequence used as a prompt for the generation. Shape is `(batch_size, sequence_length)`.
            logits_processor (LogitsProcessorList):
                A list of processors that modify the prediction scores at each generation step.
            stopping_criteria (StoppingCriteriaList):
                A list of criteria to determine when the generation loop should stop.
            generation_config (GenerationConfig):
                Configuration parameters for the generation process.
            streamer (Optional[BaseStreamer]):
                Optional streamer to stream the generated sequences.
            communicator (Optional[Union[CommunicatorBase, module]]):
                Optional communicator to use for broadcast, request, and barrier.
            logits_warper (Optional[LogitsProcessorList]):
                A list of processors used to adjust the prediction scores before multinomial sampling, required
                if `do_sample` is set to True.
            **model_kwargs:
                Additional model-specific keyword arguments to be passed to the model's `forward` method.

        Returns:
            torch.LongTensor:
                A tensor containing the generated token IDs.
        """

        my_rank = self.rank

        if my_rank == 0:
            # for the master node
            # init values
            pad_token_id = generation_config._pad_token_tensor
            has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
            do_sample = generation_config.do_sample
            if do_sample is True and not isinstance(logits_warper, LogitsProcessorList):
                raise ValueError(
                    "`do_sample` is set to `True`, `logits_warper` must be a `LogitsProcessorList` instance (it is "
                    f"{logits_warper})."
                )

            # init scores and keep track of the sequence whether it was finished
            scores = None
            unfinished = torch.ones(1, dtype=torch.long, device=input_ids.device)
            model_kwargs = self._get_initial_cache_position(input_ids, model_kwargs)

            while unfinished.item() == 1.:
                # prepare model inputs
                model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

                # forward pass to get next token
                outputs = self(**model_inputs, return_dict=True)

                # clone is needed to avoid keeping a hanging ref to outputs.logits
                # which may be very large for first iteration (the clone itself is always small)
                next_token_logits = outputs.logits[:, -1, :].clone()

                # pre-process distribution
                next_token_scores = logits_processor(input_ids, next_token_logits)
                if do_sample:
                    next_token_scores = logits_warper(input_ids, next_token_scores)

                # token selection
                if do_sample:
                    probs = nn.functional.softmax(next_token_scores, dim=-1)
                    # note: this may cause randomness during inference
                    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
                else:
                    next_tokens = torch.argmax(next_token_scores, dim=-1)

                # finished sentences should have their next token be a padding token
                if has_eos_stopping_criteria:
                    next_tokens = next_tokens * unfinished + pad_token_id * (1 - unfinished)

                # update generated ids, model inputs, and length for next step
                input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

                if streamer is not None:
                    # the current token will not be printed immediately until the last space char,
                    # which is a simple heuristic to avoid printing incomplete words.
                    streamer.put(next_tokens.cpu())

                model_kwargs = self._update_model_kwargs_for_generation(outputs, model_kwargs)

                # update finish status and synchronize with other nodes
                unfinished = unfinished & ~stopping_criteria(input_ids, scores)

                if isinstance(communicator, CommunicatorBase):
                    communicator.broadcast(int(unfinished.cpu()))
                else:  # use torch.distributed.broadcast_object_list
                    communicator.broadcast_object_list([unfinished.cpu()], src=0)

                # This is needed to properly delete outputs.logits which may be very large for first iteration
                # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
                del outputs, next_token_logits, next_token_scores

            if streamer is not None:
                streamer.end()
            return input_ids
        else:
            # for non-master nodes
            unfinished = 1
            while unfinished == 1:
                # assist forward pass to get next token
                self(**model_kwargs)
                # retrieve finish status from the master node
                if isinstance(communicator, CommunicatorBase):
                    unfinished = communicator.request()
                else:  # use torch.distributed.broadcast_object_list
                    recv_data = [torch.ones(1, dtype=torch.long)]
                    communicator.broadcast_object_list(recv_data, src=0)
                    unfinished = recv_data[0].item()

    def _validate_input(self, input_tensor):
        """
        Validates the input tensor for each node.

        Parameters:
        - input_tensor: A tensor representing the input data, expected to be of shape (batch_size, input_len).

        Raises:
        - ValueError: If the batch size is greater than 1, as currently only one user request is supported.
        - ValueError: If this is the master node, but input is not provided.
        - ValueError: If this is a non-master node, but input is provided.
        """
        my_rank = self.rank
        if my_rank == 0:
            batch_size, input_len = input_tensor.shape
            if batch_size > 1:
                raise ValueError("Currently only one user request is supported.")
            if input_len == 0:
                raise ValueError("Node with rank 0 should initiate the user request.")
        elif input_tensor not in ('', None):
            raise ValueError("Node with rank not equal to 0 should not initiate the user request.")

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        streamer: Optional["BaseStreamer"] = None,
        communicator: Optional[Union[CommunicatorBase, "module"]] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        Generates sequences of token ids for a given input.

        Args:
            inputs (Optional[torch.Tensor]):
                The input tensor to be used by the model for generation. If not provided, it defaults to None.
            generation_config (Optional[GenerationConfig]):
                The generation configuration to be used as base parametrization for the generation call.
            logits_processor (Optional[LogitsProcessorList]):
                A list of `LogitsProcessor` objects that are applied to the logits output by the model to
                modify or filter them before sampling. If not provided, an empty list is used.
            stopping_criteria (Optional[StoppingCriteriaList]):
                A list of `StoppingCriteria` objects used to determine when to stop the generation process.
                If not provided, an empty list is used.
            prefix_allowed_tokens_fn (Optional[Callable[[int, torch.Tensor], List[int]]]):
                A function that defines which tokens are allowed to follow the prefix sequence during generation.
                Useful for constrained generation scenarios. Defaults to None.
            streamer (Optional["BaseStreamer"]):
                An optional streamer object that allows for processing or outputting the generated tokens in real-time
                (e.g., streaming generation results). Defaults to None.
            communicator (Optional[Union[CommunicatorBase, module]]):
                An optional communicator object that supports broadcast, request, and barrier. Defaults to None.
            **kwargs:
                Additional keyword arguments that can include model-specific parameters or be used to
                update the `generation_config`.

        Returns:
            torch.LongTensor:
                The generated sequences as a `torch.LongTensor` containing the generated token ids.
        """
        # handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        self._validate_model_class()
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, **kwargs)
        self._validate_model_kwargs(model_kwargs.copy())

        # define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        self._validate_input(inputs_tensor)

        # kvcache
        model_kwargs["use_cache"] = generation_config.use_cache
        cache_name = "past_key_values"
        past = model_kwargs.get(cache_name, None)
        if past is None:
            model_kwargs[cache_name] = (DynamicCache())
        elif isinstance(past, tuple):
            model_kwargs[cache_name] = (DynamicCache.from_legacy_cache(past))

        my_rank = self.rank
        if my_rank == 0:
            # Pull this out first, we only use it for stopping criteria
            tokenizer = kwargs.pop("tokenizer", None)

            # execute this block only on the master node.
            # set generation parameters if not already defined.
            logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
            stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

            accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
            requires_attention_mask = "encoder_outputs" not in model_kwargs
            kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

            device = inputs_tensor.device
            self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

            # define other model kwargs.
            if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
                model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                    inputs_tensor, generation_config._pad_token_tensor, generation_config._eos_token_tensor)
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

            if generation_config.token_healing:
                input_ids = self.heal_tokens(input_ids, tokenizer)

            if streamer is not None:
                streamer.put(input_ids.cpu())

            # prepare `max_length` depending on other stopping criteria.
            input_ids_length = input_ids.shape[-1]
            has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
            has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
            generation_config = self._prepare_generated_length(
                generation_config=generation_config,
                has_default_max_length=has_default_max_length,
                has_default_min_length=has_default_min_length,
                model_input_name=model_input_name,
                inputs_tensor=inputs_tensor,
                input_ids_length=input_ids_length,
            )
            self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

            # prepare stopping criteria.
            prepared_stopping_criteria = self._get_stopping_criteria(
                generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
            )
            del tokenizer

            # prepare distribution pre_processing samplers.
            prepared_logits_processor = self._get_logits_processor(
                generation_config=generation_config,
                input_ids_seq_length=input_ids_length,
                encoder_input_ids=inputs_tensor,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                logits_processor=logits_processor,
                device=inputs_tensor.device,
                model_kwargs=model_kwargs,
            )

            # prepare logits warper.
            prepared_logits_warper = (
                self._get_logits_warper(generation_config, device=input_ids.device)
                if generation_config.do_sample else None
            )

            # run sample.
            return self._sample(
                input_ids=input_ids,
                logits_processor=prepared_logits_processor,
                logits_warper=prepared_logits_warper,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=streamer,
                communicator=communicator,
                **model_kwargs,
            )
        else:
            # for non-master nodes, just run sampling without preprocessing.
            self._sample(
                input_ids=None,
                logits_processor=None,
                logits_warper=None,
                stopping_criteria=None,
                generation_config=generation_config,
                synced_gpus=False,
                streamer=None,
                communicator=communicator,
                **model_kwargs,
            )
