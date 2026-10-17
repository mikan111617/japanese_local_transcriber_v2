"""Find pip-installed NVIDIA DLLs before importing the inference backend."""
from __future__ import annotations

import ctypes
import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

# Keep directory registrations and loaded DLLs alive for the process lifetime.
_DLL_HANDLES: list = []
_REGISTERED: set[str] = set()


def configure_dll_search() -> list[Path]:
    if sys.platform != "win32":
        return []
    roots = {Path(sysconfig.get_path(key)) for key in ("purelib", "platlib")}
    candidates = [Path(__file__).resolve().parent / "cuda" / "bin"]
    for root in sorted(roots):
        for package in ("cublas", "cuda_runtime", "cudnn", "cuda_nvrtc"):
            candidates.extend((root / "nvidia" / package / "bin",
                               root / "nvidia" / package / "lib"))
    found = [path.resolve() for path in candidates if path.is_dir()]
    for path in found:
        if str(path) not in _REGISTERED:
            _DLL_HANDLES.append(os.add_dll_directory(str(path)))
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
            _REGISTERED.add(str(path))
    return found


@dataclass(frozen=True)
class GPUStatus:
    available: bool
    detail: str
    compute_types: frozenset[str] = frozenset()


def check_gpu() -> GPUStatus:
    try:
        configure_dll_search()
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        if count < 1:
            return GPUStatus(False, "NVIDIA GPUを検出できません。GPUとドライバーを確認してください。")
        if sys.platform == "win32":
            missing = []
            # cuDNN loads its component DLLs lazily during inference. Check them
            # here too: some native loader failures otherwise terminate Python.
            for name in ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll",
                         "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
                         "cudnn_adv64_9.dll", "cudnn_graph64_9.dll",
                         "cudnn_engines_precompiled64_9.dll",
                         "cudnn_engines_runtime_compiled64_9.dll",
                         "cudnn_heuristic64_9.dll"):
                try:
                    _DLL_HANDLES.append(ctypes.WinDLL(name))
                except OSError:
                    missing.append(name)
            if missing:
                return GPUStatus(False, "不足・読み込み不可: " + ", ".join(missing)
                                 + "。install_gpu.bat を実行してアプリを再起動してください。")
        supported = frozenset(ctranslate2.get_supported_compute_types("cuda", 0))
        return GPUStatus(True, f"NVIDIA GPU {count}台を検出（GPU 0を使用）。CUDA利用可能。", supported)
    except Exception as exc:
        return GPUStatus(False, f"GPUの確認に失敗: {exc}")


def select_device(choice: str, precision: str, gpu: GPUStatus) -> tuple[str, str, str]:
    if choice == "CPU":
        return "cpu", "int8", "CPUを指定しています。"
    if not gpu.available:
        if choice == "NVIDIA GPU":
            raise RuntimeError("GPUを指定していますが利用できません。\n" + gpu.detail
                               + "\n「自動」または「CPU」を選ぶとCPUで処理できます。")
        return "cpu", "int8", "CPUで実行します。" + gpu.detail
    preferences = ("float16", "int8_float16", "float32") if precision == "自動" else (precision,)
    for compute in preferences:
        if compute in gpu.compute_types:
            return "cuda", compute, gpu.detail
    raise RuntimeError(f"このGPUは計算精度 {precision} に対応していません。「自動」を選んでください。")


def is_cuda_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(word in text for word in ("cuda", "cublas", "cudnn", "out of memory", ".dll"))


if __name__ == "__main__":
    status = check_gpu()
    print(status.detail)
    if status.available:
        print("Supported compute types:", ", ".join(sorted(status.compute_types)))
    raise SystemExit(0 if status.available else 1)
