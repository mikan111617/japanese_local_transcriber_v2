"""Inference worker. All UI communication uses events, never Tk variables."""
from __future__ import annotations

import gc
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from gpu_runtime import check_gpu, configure_dll_search, is_cuda_error, select_device
from outputs import human_duration, write_outputs

BASE_DIR = Path(__file__).resolve().parent
MODEL_OPTIONS = {
    "高精度：Whisper large-v3": "large-v3",
    "日本語特化・高速：Kotoba Whisper v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "高速：Whisper large-v3-turbo": "large-v3-turbo",
    "軽量：Whisper medium": "medium",
}


@dataclass(frozen=True)
class Job:
    files: tuple[Path, ...]
    output_dir: Path
    model_label: str
    device: str = "自動"
    precision: str = "自動"
    beam: int = 5
    vad: bool = True
    prompt: str = ""


class Cancelled(Exception):
    pass


class Transcriber:
    def __init__(self, model_factory=None, gpu_check=check_gpu):
        self.model_factory = model_factory
        self.gpu_check = gpu_check
        self.model = None
        self.model_key = None

    def _release_model(self):
        self.model = None
        self.model_key = None
        gc.collect()

    def _load(self, model_id, device, compute):
        key = (model_id, device, compute)
        if self.model_key != key:
            self._release_model()
            configure_dll_search()
            factory = self.model_factory
            if factory is None:
                from faster_whisper import WhisperModel
                factory = WhisperModel
            self.model = factory(model_id, device=device, device_index=0,
                                 compute_type=compute, download_root=str(BASE_DIR / "models"))
            self.model_key = key
        return self.model

    def run(self, job: Job, cancel: threading.Event, emit: Callable):
        completed = 0
        started = time.monotonic()
        try:
            if cancel.is_set():
                raise Cancelled
            emit("phase", "GPUと実行環境を確認しています…")
            gpu = self.gpu_check() if job.device != "CPU" else None
            device, compute, reason = select_device(job.device, job.precision, gpu)
            emit("device", (device, compute, reason))
            if job.device == "自動" and device == "cpu":
                emit("warning", reason)
            if cancel.is_set():
                raise Cancelled
            model_id = MODEL_OPTIONS[job.model_label]
            emit("phase", "モデルを準備中です。初回はダウンロードに時間がかかります…")
            try:
                self._load(model_id, device, compute)
            except Exception as exc:
                if device != "cuda" or job.device != "自動" or not is_cuda_error(exc):
                    raise
                device, compute = self._fallback(model_id, exc, emit)
            for index, audio in enumerate(job.files):
                if cancel.is_set():
                    raise Cancelled
                emit("file", (index, "処理中"))
                emit("preview", (index, ""))
                emit("phase", f"{audio.name} を解析しています…")
                try:
                    segments, info, duration = self._transcribe(job, audio, index, model_id, cancel, emit)
                except Exception as exc:
                    if device != "cuda" or job.device != "自動" or not is_cuda_error(exc):
                        raise
                    device, compute = self._fallback(model_id, exc, emit)
                    # Discard partial GPU output before retrying this file.
                    emit("preview", (index, ""))
                    emit("progress", index / len(job.files) * 100)
                    segments, info, duration = self._transcribe(job, audio, index, model_id, cancel, emit)
                if cancel.is_set():
                    raise Cancelled
                metadata = {
                    "model": model_id, "model_label": job.model_label,
                    "device": device, "compute_type": compute, "language": "ja",
                    "language_probability": getattr(info, "language_probability", None),
                    "duration_seconds": round(duration, 3), "beam_size": job.beam,
                    "vad_filter": job.vad, "initial_prompt": job.prompt or None,
                }
                outputs = write_outputs(audio, job.output_dir, segments, metadata)
                completed += 1
                emit("saved", (index, outputs))
                emit("file", (index, "完了"))
                emit("log", f"保存完了: {outputs['TXT']}")
                emit("progress", completed / len(job.files) * 100)
            emit("finished", f"{completed}件の文字起こしが完了しました（{human_duration(time.monotonic() - started)}）。")
        except Cancelled:
            emit("cancelled", f"中止しました。{completed}件は保存済みです。処理途中のファイルは保存していません。")
        except Exception as exc:
            emit("log", traceback.format_exc())
            emit("error", f"{exc}\n\n保存済み: {completed}件。詳細は「処理ログ」で確認できます。")

    def _fallback(self, model_id, exc, emit):
        reason = f"GPUでエラーが発生したためCPUで再試行します: {exc}"
        emit("warning", reason)
        emit("phase", "CPU用のモデルを準備しています…")
        self._load(model_id, "cpu", "int8")
        emit("device", ("cpu", "int8", reason))
        return "cpu", "int8"

    def _transcribe(self, job, audio, index, model_id, cancel, emit):
        kotoba = "kotoba-whisper" in model_id
        iterator, info = self.model.transcribe(
            str(audio), language="ja", task="transcribe", beam_size=job.beam,
            vad_filter=job.vad,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 200} if job.vad else None,
            initial_prompt=job.prompt or None, condition_on_previous_text=not kotoba,
            word_timestamps=False, temperature=0.0, compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0, no_speech_threshold=0.6, chunk_length=15 if kotoba else 30,
        )
        duration = float(getattr(info, "duration", 0) or 0)
        segments = []
        for seg in iterator:
            if cancel.is_set():
                raise Cancelled
            segments.append({
                "start": round(float(seg.start), 3), "end": round(float(seg.end), 3),
                "text": seg.text, "avg_logprob": round(float(seg.avg_logprob), 4),
                "no_speech_prob": round(float(seg.no_speech_prob), 4),
            })
            emit("segment", (index, seg.text.strip()))
            local = min(1.0, seg.end / duration) if duration else 0
            emit("progress", (index + local) / len(job.files) * 100)
            emit("status", f"{index + 1}/{len(job.files)}件 · {audio.name} · "
                 f"{human_duration(seg.end)} / {human_duration(duration)}")
        return segments, info, duration
