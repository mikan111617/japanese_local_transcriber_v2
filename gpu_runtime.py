"""Find pip-installed NVIDIA DLLs before importing the inference backend."""
from __future__ import annotations

import ctypes
import os
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# cuDNN loads its component DLLs lazily during inference. Check them up front
# too: some native loader failures otherwise terminate Python.
REQUIRED_DLLS = ("cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll",
                 "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll",
                 "cudnn_adv64_9.dll", "cudnn_graph64_9.dll",
                 "cudnn_engines_precompiled64_9.dll",
                 "cudnn_engines_runtime_compiled64_9.dll",
                 "cudnn_heuristic64_9.dll")

# Keep directory registrations and loaded DLLs alive for the process lifetime.
_DLL_HANDLES: list = []
_REGISTERED: set[str] = set()


def _package_roots() -> list[Path]:
    roots = [sysconfig.get_path(key) for key in ("purelib", "platlib")]
    roots += [entry for entry in sys.path if entry]
    # install_gpu.bat installs into the app's .venv. Its DLLs also work when
    # app.py was started with another Python.
    roots.append(APP_DIR / ".venv" / "Lib" / "site-packages")
    return [Path(root) for root in roots]


def configure_dll_search() -> list[Path]:
    if sys.platform != "win32":
        return []
    candidates = [APP_DIR / "cuda" / "bin"]
    for root in _package_roots():
        for pattern in ("*/bin", "*/lib"):
            candidates.extend(sorted((root / "nvidia").glob(pattern)))
    found = []
    for path in candidates:
        if path.is_dir() and path.resolve() not in found:
            found.append(path.resolve())
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
    # Not-found DLLs, and extra lines for the log and install_gpu.bat output.
    missing: tuple[str, ...] = ()
    diagnostics: str = ""


def _load_error(exc: OSError) -> str:
    code = getattr(exc, "winerror", None)
    if code == 1455:
        return "仮想メモリ（ページングファイル）不足"
    if code == 126:
        return "依存するDLLを読み込めません"
    if code == 193:
        return "64bit版ではないか、ファイルが壊れています"
    return str(exc)


def _load_runtime_dlls() -> tuple[list[str], list[str]]:
    path_dirs = [Path(entry.strip('"')) for entry in os.environ.get("PATH", "").split(os.pathsep)
                 if entry.strip('"')]
    missing, failed = [], []
    for name in REQUIRED_DLLS:
        try:
            _DLL_HANDLES.append(ctypes.WinDLL(name))
            continue
        except OSError:
            pass
        # A bare name skips PATH. Load the file directly, which also reports
        # why a DLL that is present cannot be loaded.
        path = next((folder / name for folder in path_dirs if (folder / name).is_file()), None)
        if path is None:
            missing.append(name)
            continue
        try:
            _DLL_HANDLES.append(ctypes.WinDLL(str(path)))
        except OSError as exc:
            failed.append(f"{name}（{_load_error(exc)}）")
    return missing, failed


def _dll_problem(dll_dirs: list[Path], missing: list[str], failed: list[str]) -> GPUStatus:
    lines = [f"Python: {sys.executable}",
             "DLLフォルダー: " + (", ".join(map(str, dll_dirs)) or "なし")]
    lines += [f"見つからない: {name}" for name in missing]
    lines += [f"読み込み失敗: {item}" for item in failed]
    if missing and not dll_dirs:
        detail = ("NVIDIAランタイム（cuBLAS・cuDNN）がこのPython環境に見つかりません。"
                  "アプリを閉じて install_gpu.bat を実行し、最後に「GPU setup complete」と"
                  "表示されたことを確認してから run.bat で起動してください。")
    elif missing:
        detail = ("不足: " + ", ".join(missing) + "。インストールが途中で止まった可能性があります。"
                  "アプリを閉じて install_gpu.bat を再実行してください。")
    else:
        detail = "読み込み不可: " + ", ".join(failed) + "。"
        if any("ページング" in item for item in failed):
            detail += "Windowsの仮想メモリを増やすか、他のアプリを閉じてから再起動してください。"
        else:
            detail += "NVIDIAドライバーを更新してPCを再起動し、install_gpu.bat を再実行してください。"
    return GPUStatus(False, detail + "詳細は「処理ログ」を確認してください。",
                     missing=tuple(missing), diagnostics="\n".join(lines))


def check_gpu() -> GPUStatus:
    try:
        dll_dirs = configure_dll_search()
        import ctranslate2
        count = ctranslate2.get_cuda_device_count()
        if count < 1:
            return GPUStatus(False, "NVIDIA GPUを検出できません。GPUとドライバーを確認してください。")
        if sys.platform == "win32":
            missing, failed = _load_runtime_dlls()
            if missing or failed:
                return _dll_problem(dll_dirs, missing, failed)
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
    if status.diagnostics:
        print(status.diagnostics)
    if status.available:
        print("Supported compute types:", ", ".join(sorted(status.compute_types)))
    # Exit code 2 tells install_gpu.bat that reinstalling the DLLs may help.
    raise SystemExit(0 if status.available else 2 if status.missing else 1)
