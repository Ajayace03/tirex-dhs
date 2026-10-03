"""Process setup, model loading, and the cost measurements (time, FLOPs, memory)."""
from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np


def configure_process(log_path: Path, threads: int) -> logging.Logger:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
    import torch

    torch.set_num_threads(threads)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tirex")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S")
    for handler in (logging.FileHandler(log_path, mode="w", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def allow_native_kernels() -> None:
    """Select TiRex's PyTorch kernels when Triton is missing or its DLL is blocked."""
    try:
        import triton  # noqa: F401
        return
    except Exception:
        for name in [key for key in sys.modules if key == "triton" or key.startswith("triton.")]:
            del sys.modules[name]

    class _TritonAsMissing:
        def find_spec(self, fullname, path, target=None):
            if fullname == "triton" or fullname.startswith("triton."):
                raise ModuleNotFoundError(fullname)
            return None

    sys.meta_path.insert(0, _TritonAsMissing())


def load_backbone(source: str, device: str):
    allow_native_kernels()
    from tirex2 import load_model

    # TiRex-2's CUDA kernels need compiled Triton/flashrnn builds. Its pure-PyTorch kernels,
    # selected by loading for "cpu", run on any device, so load that way and then move.
    loaded = load_model(source, device="cpu")
    backbone = loaded.model.to(device)
    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    backbone.eval()
    return backbone, np.asarray(loaded.quantiles, dtype=np.float64)


def synchronize(device: str) -> None:
    """Wait for queued GPU work so wall-clock timings are honest."""
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def gpu_peak_mb(device: str) -> float:
    if not device.startswith("cuda"):
        return 0.0
    import torch

    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def reset_gpu_peak(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()


def working_set_mb() -> float:
    """Resident memory of this process in MB (Windows working set, or ru_maxrss elsewhere)."""
    if os.name != "nt":
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    class _Counters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel.K32GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
    query.restype = ctypes.c_int
    counters = _Counters()
    counters.cb = ctypes.sizeof(_Counters)
    if not query(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        return 0.0
    return counters.WorkingSetSize / (1024 * 1024)


class MemorySampler:
    """Samples resident memory every few milliseconds; peak_mb is the highest reading."""

    def __init__(self, interval: float = 0.002):
        self.interval = interval
        self.baseline_mb = working_set_mb()
        self.peak_mb = self.baseline_mb
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, working_set_mb())
            time.sleep(self.interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak_mb = max(self.peak_mb, working_set_mb())

    @property
    def added_mb(self) -> float:
        return self.peak_mb - self.baseline_mb


def count_flops(fn) -> float:
    """Floating point operations of one call, from PyTorch's operator-level counter."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    with torch.no_grad(), FlopCounterMode(display=False) as counter:
        fn()
    return float(counter.get_total_flops())
