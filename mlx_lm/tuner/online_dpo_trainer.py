# Copyright © 2024 Apple Inc.

from dataclasses import dataclass, field
from pathlib import Path
import time
from abc import ABC, abstractmethod
from typing import List, Union, Optional, Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx.nn.utils import average_gradients
from mlx.utils import tree_flatten

from .trainer import grad_checkpoint, TrainingArgs


@dataclass
class OnlineDPOTrainingArgs(TrainingArgs):
    beta: float = field(
        default=0.1, metadata={"help": "Temperature parameter for DPO training."}
    )
    loss_type: str = field(
        default="sigmoid",
        metadata={"help": "DPO loss type: 'sigmoid', 'hinge', 'ipo', or 'dpop'."},
    )
    delta: float = field(
        default=50.0, metadata={"help": "Delta parameter for DPOP loss type."}
    )
    judge: str = field(
        default=None,
        metadata={
            "help": "Judge. If None, can only be (PairRMJudge, human) default is PairRMJudge."
        },
    )


class Judge(ABC):
    def __init__(self, mode: str = "rank"):
        """
        Initialize the judge.
        
        Args:
            mode (`str`, *optional*, defaults to `"rank"`):
                The mode of the judge. Can be "rank", "pairwise", or "pairrm".
        """
        self.mode = mode
        self.blender = None
        
        # Initialize PairRM if needed
        if mode == "pairrm":
            self._init_pairrm()
    
    def _init_pairrm(self):
        """Initialize the PairRM model if needed."""
        try:
            import llm_blender
            from accelerate import Accelerator
            self.blender = llm_blender.Blender()
            self.blender.loadranker("llm-blender/PairRM", device=Accelerator().device)
        except ImportError:
            raise ValueError("llm-blender is not installed. Please install it with `pip install llm-blender`.")
    
    @abstractmethod
    def _judge_impl(self, prompts: List[str], completions: List[List[str]], 
                   shuffle_order: bool = True, **kwargs) -> List[Any]:
        """
        Implementation method to be overridden by subclasses.
        """
        raise NotImplementedError("Judge subclasses must implement the `_judge_impl` method.")
    
    def judge(self, prompts: List[str], completions: List[List[str]], 
             shuffle_order: bool = True, mode: Optional[str] = None, **kwargs) -> List[Any]:
        """
        Judge the completions for the given prompts.
        
        Args:
            prompts (`List[str]`):
                List of prompts.
            completions (`List[List[str]]`):
                List of completions for each prompt.
            shuffle_order (`bool`, *optional*, defaults to `True`):
                Whether to shuffle the order of the completions to avoid positional bias.
            mode (`str`, *optional*):
                Override the judge mode for this call. Can be "rank", "pairwise", or "pairrm".
            **kwargs:
                Additional arguments specific to the mode.
                For "pairrm" mode:
                - return_scores (`bool`): Return probability scores instead of ranks.
                - temperature (`float`): Temperature for scaling logits if return_scores is True.
        
        Returns:
            For "rank" mode:
                `List[List[int]]`: List of lists of indices, where each list contains the ranks of the completions.
            For "pairwise" mode:
                `List[int]`: List of indices indicating the preferred completion (0 or 1) for each prompt.
            For "pairrm" mode with return_scores=False:
                `List[int]`: List of indices indicating the preferred completion (0 or 1) for each prompt.
            For "pairrm" mode with return_scores=True:
                `List[float]`: List of probability scores for the first completion.
        """
        # Use provided mode or fall back to instance mode
        current_mode = mode if mode is not None else self.mode
        
        if current_mode == "pairrm":
            return self._judge_pairrm(prompts, completions, shuffle_order, **kwargs)
        elif hasattr(self, "_judge_impl"):
            return self._judge_impl(prompts, completions, shuffle_order, mode=current_mode, **kwargs)
        else:
            raise NotImplementedError(f"No implementation available for mode: {current_mode}")
    
    def _judge_pairrm(self, prompts: List[str], completions: List[List[str]], 
                     shuffle_order: bool = True, return_scores: bool = False, 
                     temperature: float = 1.0) -> List[Union[int, float]]:
        """
        Judge using the PairRM model.
        """
        if self.blender is None:
            self._init_pairrm()
            
        if any(len(pair) != 2 for pair in completions):
            raise ValueError("PairRM judge requires exactly 2 completions per prompt.")
            
        # Shuffle the order of the completions to avoid positional bias
        flip_mask = None
        if shuffle_order:
            flip_mask = np.random.choice([True, False], size=len(prompts))
            completions = [pair[::-1] if flip else pair for flip, pair in zip(flip_mask, completions)]

        # Rank the completions
        ranks = self.blender.rank(prompts, completions, return_scores=return_scores, disable_tqdm=True)
        if not return_scores:
            ranks -= 1  # PairRM rank is 1-indexed, so we subtract 1 to make it 0-indexed
        else:
            # scale the logits by temperature
            ranks /= temperature

        # Flip back the ranks or scores to the original order if needed
        if shuffle_order and flip_mask is not None:
            if return_scores:
                ranks[flip_mask] = ranks[flip_mask][:, ::-1]
            else:
                # For non-score ranks, we need to invert the result for flipped pairs
                for i, flipped in enumerate(flip_mask):
                    if flipped:
                        ranks[i] = 1 - ranks[i]

        # Return the ranks or score probability
        if return_scores:
            logit_max = np.amax(ranks, axis=-1, keepdims=True)
            exp_logit_shifted = np.exp(ranks - logit_max)
            probs = exp_logit_shifted / np.sum(exp_logit_shifted, axis=-1, keepdims=True)
            return probs[:, 0].tolist()
        else:
            return ranks.tolist()


def default_loss(model, batch, lengths):
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    logits = model(inputs)
    logits = logits.astype(mx.float32)

    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    ce = ce.sum() / ntoks

    return ce, ntoks


def iterate_batches(
    dataset,
    tokenizer,
    batch_size,
    max_seq_length,
    train=False,
):
    # Sort by length:
    idx = sorted(range(len(dataset)), key=lambda idx: len(dataset[idx]))
    if len(dataset) < batch_size:
        raise ValueError(
            f"Dataset must have at least batch_size={batch_size}"
            f" examples but only has {len(dataset)}."
        )

    # If running in distributed mode (N machines) then each one should skip N-1
    # samples
    step = mx.distributed.init().size()
    if batch_size % step != 0:
        raise ValueError("The batch size must be divisible by the number of workers")

    # Make the batches:
    batch_idx = [
        idx[i : i + batch_size : step]
        for i in range(0, len(idx) - batch_size + 1, batch_size)
    ]

    while True:
        indices = np.random.permutation(len(batch_idx))
        for i in indices:
            batch = [dataset[j] for j in batch_idx[i]]
            if len(batch[0]) == 2:
                batch, offsets = zip(*batch)
            else:
                offsets = [0] * len(batch)
            lengths = [len(x) for x in batch]
            if max(lengths) > max_seq_length:
                print(
                    f"[WARNING] Some sequences are longer than {max_seq_length} tokens. "
                    f"The longest sentence {max(lengths)} will be truncated to {max_seq_length}. "
                    "Consider pre-splitting your data to save memory."
                )

            # Pad to the nearest multiple of 8 or the maximum length
            pad_to = 8
            max_length_in_batch = pad_to * ((max(lengths) + pad_to - 1) // pad_to)
            max_length_in_batch = min(max_length_in_batch, max_seq_length)

            batch_arr = np.zeros((batch_size // step, max_length_in_batch), np.int32)

            for j in range(batch_size // step):
                truncated_length = min(lengths[j], max_seq_length)
                batch_arr[j, :truncated_length] = batch[j][:truncated_length]
                lengths[j] = (
                    truncated_length  # Update lengths to match truncated lengths
                )
            batch = mx.array(batch_arr)
            yield batch, mx.array(list(zip(offsets, lengths)))

        if not train:
            break


def evaluate(
    model,
    dataset,
    tokenizer,
    batch_size,
    num_batches,
    max_seq_length=2048,
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
):
    all_losses = mx.array(0.0)
    ntokens = mx.array(0)

    index_iterator = iter(range(num_batches)) if num_batches != -1 else iter(int, 1)

    for _, batch in zip(
        index_iterator,
        iterate_batches(
            dataset=dataset,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
        ),
    ):
        losses, toks = loss(model, *batch)
        all_losses += losses * toks
        ntokens += toks
        mx.eval(all_losses, ntokens)

    all_losses = mx.distributed.all_sum(all_losses, stream=mx.cpu)
    ntokens = mx.distributed.all_sum(ntokens, stream=mx.cpu)

    return (all_losses / ntokens).item()


class TrainingCallback:

    def on_train_loss_report(self, train_info: dict):
        """Called to report training loss at specified intervals."""
        pass

    def on_val_loss_report(self, val_info: dict):
        """Called to report validation loss at specified intervals or the beginning."""
        pass


def train(
    model,
    tokenizer,
    optimizer,
    train_dataset,
    val_dataset,
    args: TrainingArgs = TrainingArgs(),
    loss: callable = default_loss,
    iterate_batches: callable = iterate_batches,
    training_callback: TrainingCallback = None,
):
    print(f"Starting training..., iters: {args.iters}")
    world = mx.distributed.init()
    world_size = world.size()
    rank = world.rank()
    if world_size > 1:
        print(f"Node {rank} of {world_size}")

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    state = [model.state, optimizer.state]

    def step(batch):
        # Forward and backward pass
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)

        # All reduce the gradients if running in distributed mode
        grad = average_gradients(grad)

        # Model update
        optimizer.update(model, grad)

        return lvalue, toks

    loss_value_and_grad = nn.value_and_grad(model, loss)

    losses = 0
    n_tokens = 0
    steps = 0
    trained_tokens = 0
    train_time = 0
    # Main training loop
    for it, batch in zip(
        range(1, args.iters + 1),
        iterate_batches(
            dataset=train_dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            train=True,
        ),
    ):
        tic = time.perf_counter()
        # Report validation loss if needed, the first validation loss
        # is always measured before any training.
        if it == 1 or it % args.steps_per_eval == 0 or it == args.iters:
            tic = time.perf_counter()
            val_loss = evaluate(
                model=model,
                dataset=val_dataset,
                loss=loss,
                tokenizer=tokenizer,
                batch_size=args.batch_size,
                num_batches=args.val_batches,
                max_seq_length=args.max_seq_length,
                iterate_batches=iterate_batches,
            )
            val_time = time.perf_counter() - tic
            if rank == 0:
                print(
                    f"Iter {it}: "
                    f"Val loss {val_loss:.3f}, "
                    f"Val took {val_time:.3f}s",
                    flush=True,
                )

            if training_callback is not None:
                val_info = {
                    "iteration": it,
                    "val_loss": val_loss,
                    "val_time": val_time,
                }
                training_callback.on_val_loss_report(val_info)

            tic = time.perf_counter()

        lvalue, toks = step(batch)
        losses += lvalue
        n_tokens += toks
        steps += 1
        mx.eval(state, losses, n_tokens)
        train_time += time.perf_counter() - tic

        # Report training loss if needed
        if it % args.steps_per_report == 0 or it == args.iters:
            train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
            train_loss /= steps * mx.distributed.init().size()
            n_tokens = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
            learning_rate = optimizer.learning_rate.item()
            it_sec = args.steps_per_report / train_time
            tokens_sec = float(n_tokens) / train_time
            trained_tokens += n_tokens
            peak_mem = mx.get_peak_memory() / 1e9
            if rank == 0:
                print(
                    f"Iter {it}: Train loss {train_loss:.3f}, "
                    f"Learning Rate {learning_rate:.3e}, "
                    f"It/sec {it_sec:.3f}, "
                    f"Tokens/sec {tokens_sec:.3f}, "
                    f"Trained Tokens {trained_tokens}, "
                    f"Peak mem {peak_mem:.3f} GB",
                    flush=True,
                )

            if training_callback is not None:
                train_info = {
                    "iteration": it,
                    "train_loss": train_loss,
                    "learning_rate": learning_rate,
                    "iterations_per_second": it_sec,
                    "tokens_per_second": tokens_sec,
                    "trained_tokens": trained_tokens,
                    "peak_memory": peak_mem,
                }
                training_callback.on_train_loss_report(train_info)

            losses = 0
            n_tokens = 0
            steps = 0
            train_time = 0

        # Save adapter weights
        if it % args.steps_per_save == 0:
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(args.adapter_file), adapter_weights)
            checkpoint = (
                Path(args.adapter_file).parent / f"{it:07d}_adapters.safetensors"
            )
            mx.save_safetensors(str(checkpoint), adapter_weights)
            print(
                f"Iter {it}: Saved adapter weights to "
                f"{args.adapter_file} and {checkpoint}."
            )

    # Save final weights
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(args.adapter_file), adapter_weights)
    print(f"Saved final weights to {args.adapter_file}.")
