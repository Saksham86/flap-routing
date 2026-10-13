"""GPU energy measurement via NVML.

Two independent methods, used together so they can cross-check each other:
  * counter: nvmlDeviceGetTotalEnergyConsumption (mJ since driver load; Volta and newer).
  * sampler: a background thread polls nvmlDeviceGetPowerUsage (mW) and integrates
             with the trapezoid rule.
If NVML is missing (CPU-only machine, no driver) the meter reports ``available=False``
with a reason instead of raising, so the same scripts run on edge CPUs.

Note: NVML's device index can differ from CUDA's when CUDA_VISIBLE_DEVICES is set.
On Colab (one GPU) both are 0.
"""
from __future__ import annotations

import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class EnergyReading:
    available: bool
    duration_s: float
    counter_joules: float | None = None
    sampled_joules: float | None = None
    mean_power_w: float | None = None
    max_power_w: float | None = None
    n_samples: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class EnergyMeter:
    def __init__(self, gpu_index: int = 0, sample_interval_s: float = 0.02,
                 use_counter: bool = True, use_sampler: bool = True):
        self.gpu_index = gpu_index
        self.sample_interval_s = sample_interval_s
        self.use_counter = use_counter
        self.use_sampler = use_sampler
        self.available = False
        self.reason: str | None = None
        self.counter_supported = False
        self._nvml = None
        self._handle = None
        self._samples: list[tuple[float, float]] = []
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._e0_mj: int | None = None
        self._notes: list[str] = []
        self._init_nvml()

    # ---- setup -----------------------------------------------------------------------
    def _init_nvml(self) -> None:
        try:
            import pynvml
        except ImportError:
            self.reason = "pynvml (nvidia-ml-py) not installed"
            return
        try:
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
        except Exception as e:  # NVMLError_LibraryNotFound, DriverNotLoaded, ...
            self.reason = f"NVML unavailable: {e}"
            return
        self._nvml = pynvml
        self.available = True
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            self.counter_supported = True
        except Exception as e:
            self._notes.append(f"energy counter not supported: {e}")

    def device_name(self) -> str | None:
        if not self.available:
            return None
        name = self._nvml.nvmlDeviceGetName(self._handle)
        return name.decode() if isinstance(name, bytes) else name

    def power_w(self) -> float | None:
        if not self.available:
            return None
        return self._nvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0

    # ---- measurement -----------------------------------------------------------------
    def _sample_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                self._samples.append((time.perf_counter(), self.power_w()))
            except Exception as e:  # keep measuring; record the problem once
                if not any("sample error" in n for n in self._notes):
                    self._notes.append(f"sample error: {e}")
            self._stop_evt.wait(self.sample_interval_s)

    def start(self) -> "EnergyMeter":
        self._samples = []
        self._stop_evt.clear()
        self._t0 = time.perf_counter()
        if self.available and self.use_counter and self.counter_supported:
            self._e0_mj = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
        if self.available and self.use_sampler:
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> EnergyReading:
        t1 = time.perf_counter()
        duration = t1 - self._t0
        if not self.available:
            return EnergyReading(available=False, duration_s=duration, notes=[self.reason or "unavailable"])
        counter_j = None
        if self._e0_mj is not None:
            e1 = self._nvml.nvmlDeviceGetTotalEnergyConsumption(self._handle)
            counter_j = (e1 - self._e0_mj) / 1000.0
            self._e0_mj = None
        if self._thread is not None:
            self._stop_evt.set()
            self._thread.join(timeout=2.0)
            self._thread = None
            try:  # closing sample so the integral covers the full window
                self._samples.append((t1, self.power_w()))
            except Exception:
                pass
        sampled_j = mean_w = max_w = None
        s = [(t, p) for t, p in self._samples if p is not None]
        if len(s) >= 2:
            sampled_j = sum((t2 - t1_) * (p1 + p2) / 2.0 for (t1_, p1), (t2, p2) in zip(s, s[1:]))
            span = s[-1][0] - s[0][0]
            mean_w = sampled_j / span if span > 0 else None
            max_w = max(p for _, p in s)
        notes = list(self._notes)
        if duration < 1.0:
            notes.append("window < 1 s: NVML counters/power update coarsely; prefer longer windows")
        return EnergyReading(available=True, duration_s=duration, counter_joules=counter_j,
                             sampled_joules=sampled_j, mean_power_w=mean_w, max_power_w=max_w,
                             n_samples=len(s), notes=notes)

    def measure_idle(self, seconds: float = 3.0) -> EnergyReading:
        self.start()
        time.sleep(seconds)
        return self.stop()

    def __enter__(self) -> "EnergyMeter":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.last = self.stop()

    def close(self) -> None:
        if self.available:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
            self.available = False
