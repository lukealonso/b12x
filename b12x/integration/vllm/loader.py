"""GB10 checkpoint loading into persistent CPU-addressable CUDA allocations."""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader
from vllm.model_executor.model_loader.weight_utils import _BAR_FORMAT, enable_tqdm

from b12x.loader import storage_stats
from b12x.loader._checkpoint import DirectWeightSession
from b12x.loader._pool import owns_storage, owns_tensor, weight_pool

logger = init_logger("vllm.model_executor.model_loader.b12x")


class B12xModelLoader(DefaultModelLoader):
    """Route checkpoint metadata, then O_DIRECT-read into final pinned storage."""

    def __init__(self, load_config):
        options = dict(load_config.model_loader_extra_config)
        self.io_threads = options.pop("io_threads", 8)
        if options.get("enable_multithread_load"):
            raise ValueError("b12x currently uses synchronous checkpoint routing")
        if load_config.safetensors_load_strategy not in (None, "lazy"):
            raise ValueError(
                "b12x O_DIRECT input does not use safetensors read strategies"
            )
        super().__init__(
            dataclasses.replace(
                load_config,
                load_format="safetensors",
                model_loader_extra_config=options,
            )
        )
        self._session = None

    def load_model(self, vllm_config, model_config, prefix=""):
        from vllm.model_executor.weight_transfer import weight_transfer

        if model_config.enable_sleep_mode:
            raise ValueError("b12x shared allocations do not support vLLM sleep mode")
        device = torch.device(
            self.load_config.device or vllm_config.device_config.device
        )
        if device.type != "cuda":
            raise ValueError("the initial b12x loader requires a CUDA device")
        index = torch.cuda.current_device() if device.index is None else device.index
        with (
            weight_pool(allocation="pinned_wc", device=index) as allocator,
            DirectWeightSession(
                index, io_threads=self.io_threads, allocation_scope=allocator
            ) as session,
            weight_transfer(session, allocator=allocator),
        ):
            self._session = session
            try:
                model = super().load_model(vllm_config, model_config, prefix)
                io_stats = session.stats()
                shared_runtime_buffers = [
                    f"{module_name}.{name}".lstrip(".")
                    for module_name, module in model.named_modules()
                    for name, buffer in module.named_buffers(recurse=False)
                    if name in module._non_persistent_buffers_set
                    and owns_storage(buffer)
                ]
                if shared_runtime_buffers:
                    raise RuntimeError(
                        "runtime buffers were allocated in shared weight storage: "
                        + ", ".join(shared_runtime_buffers)
                    )
            finally:
                self._session = None
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        parameter_bytes = sum(p.nbytes for p in model.parameters())
        shared_bytes = sum(p.nbytes for p in model.parameters() if owns_tensor(p))
        model._b12x_loader_storage = {
            "allocation": "pinned_wc",
            "parameter_bytes": parameter_bytes,
            "shared_parameter_bytes": shared_bytes,
            "shared_runtime_buffers": shared_runtime_buffers,
            **storage_stats(),
            "io": io_stats,
        }
        logger.info("b12x O_DIRECT I/O counters: %s", io_stats)
        logger.info("b12x allocation audit: no shared non-persistent runtime buffers")
        logger.info(
            "b12x final parameters: %.3f / %.3f GiB in write-combined shared storage; "
            "pool backing %.3f GiB",
            shared_bytes / 2**30,
            parameter_bytes / 2**30,
            storage_stats()["live_bytes"] / 2**30,
        )
        return model

    def load_weights(self, model, model_config):
        super().load_weights(model, model_config)
        if self._session is not None:
            self._session.flush()

    @staticmethod
    def _needs_values(entry):
        return math.prod(entry.shape) == 1 or entry.name.rsplit(".", 1)[-1] in {
            "layer_multipliers",
            "ngram_heads_offsets",
            "ngram_heads_vocab_sizes",
        }

    def _get_weights_iterator(self, source):
        from vllm.model_executor.model_loader.ep_weight_filter import should_skip_weight

        if self._session is None:
            raise RuntimeError("b12x requires an active initial-load session")
        folder, files, _ = self._prepare_weights(
            source.model_or_path,
            source.subfolder,
            source.revision,
            False,
            source.allow_patterns_overrides,
            source.weight_name_prefixes,
        )
        if self.counter_before_loading_weights == 0.0:
            self.counter_before_loading_weights = time.perf_counter()
        logger.info("b12x O_DIRECT input: %d safetensors shards", len(files))
        with tqdm(
            files,
            desc="Loading safetensors checkpoint shards (b12x)",
            disable=not enable_tqdm(self.load_config.use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        ) as shards:
            yield from self._session.weights(
                shards,
                prefixes=source.weight_name_prefixes,
                prefix=source.prefix,
                index_path=Path(folder) / "model.safetensors.index.json",
                needs_values=self._needs_values,
                skip=lambda name: should_skip_weight(name, self.local_expert_ids),
            )


def register_b12x_loader():
    from vllm.model_executor.model_loader import register_model_loader

    register_model_loader("b12x")(B12xModelLoader)
