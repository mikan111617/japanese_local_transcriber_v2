
from __future__ import annotations

import json
import os
import queue
import threading
import traceback
import ctypes
from pathlib import Path
from tkinter import (
    Tk, StringVar, BooleanVar, END, NORMAL, DISABLED, HORIZONTAL,
    filedialog, messagebox
)
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

from faster_whisper import WhisperModel
import ctranslate2


APP_TITLE = "日本語ローカル文字起こし"
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_OPTIONS = {
    "高精度：Whisper large-v3": "large-v3",
    "日本語特化・高速：Kotoba Whisper v2.0": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "高速：Whisper large-v3-turbo": "large-v3-turbo",
    "軽量：Whisper medium": "medium",
}

AUDIO_TYPES = [
    ("音声・動画", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.opus *.mp4 *.mov *.mkv *.webm"),
    ("すべてのファイル", "*.*"),
]


def srt_timestamp(seconds: float) -> str:
    ms = max(0, int(round(seconds * 1000)))
    hours, ms = divmod(ms, 3_600_000)
    minutes, ms = divmod(ms, 60_000)
    secs, ms = divmod(ms, 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{ms:03}"


def vtt_timestamp(seconds: float) -> str:
    return srt_timestamp(seconds).replace(",", ".")


def human_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}時間{m}分{s}秒"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def write_outputs(
    audio_path: Path,
    output_dir: Path,
    segments: list[dict],
    metadata: dict
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = audio_path.stem

    plain_text = "".join(seg["text"].strip() for seg in segments).strip()
    timestamped_text = "\n".join(
        f"[{srt_timestamp(seg['start'])} - {srt_timestamp(seg['end'])}] {seg['text'].strip()}"
        for seg in segments
    )

    txt_path = output_dir / f"{stem}.txt"
    timestamped_path = output_dir / f"{stem}_timestamps.txt"
    srt_path = output_dir / f"{stem}.srt"
    vtt_path = output_dir / f"{stem}.vtt"
    json_path = output_dir / f"{stem}.json"

    txt_path.write_text(plain_text, encoding="utf-8-sig")
    timestamped_path.write_text(timestamped_text, encoding="utf-8-sig")

    srt_lines = []
    for idx, seg in enumerate(segments, 1):
        srt_lines.extend([
            str(idx),
            f"{srt_timestamp(seg['start'])} --> {srt_timestamp(seg['end'])}",
            seg["text"].strip(),
            "",
        ])
    srt_path.write_text("\n".join(srt_lines), encoding="utf-8-sig")

    vtt_lines = ["WEBVTT", ""]
    for seg in segments:
        vtt_lines.extend([
            f"{vtt_timestamp(seg['start'])} --> {vtt_timestamp(seg['end'])}",
            seg["text"].strip(),
            "",
        ])
    vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")

    payload = {
        "source_file": str(audio_path),
        **metadata,
        "text": plain_text,
        "segments": segments,
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return {
        "TXT": txt_path,
        "タイムスタンプTXT": timestamped_path,
        "SRT": srt_path,
        "VTT": vtt_path,
        "JSON": json_path,
    }


class TranscriberApp:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("920x760")
        self.root.minsize(820, 680)

        self.files: list[Path] = []
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.ui_queue: queue.Queue = queue.Queue()
        self.loaded_model: WhisperModel | None = None
        self.loaded_model_key: tuple | None = None

        self.model_var = StringVar(value="高精度：Whisper large-v3")
        self.device_var = StringVar(value="自動")
        self.beam_var = StringVar(value="5")
        self.vad_var = BooleanVar(value=True)
        self.output_var = StringVar(value=str(BASE_DIR / "transcriptions"))
        self.status_var = StringVar(value="音声ファイルを選択してください。")

        self._build_ui()
        self.root.after(100, self._drain_ui_queue)

    def _build_ui(self):
        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)

        title = ttk.Label(
            outer,
            text="日本語ローカル文字起こし",
            font=("", 18, "bold")
        )
        title.pack(anchor="w")

        desc = ttk.Label(
            outer,
            text="音声/動画をローカルで文字起こしします。初回のみモデルをダウンロードし、以後はローカルで利用できます。"
        )
        desc.pack(anchor="w", pady=(4, 12))

        file_frame = ttk.LabelFrame(outer, text="1. 入力ファイル", padding=10)
        file_frame.pack(fill="x")

        button_row = ttk.Frame(file_frame)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="ファイルを選択", command=self._choose_files).pack(side="left")
        ttk.Button(button_row, text="一覧をクリア", command=self._clear_files).pack(side="left", padx=(8, 0))

        self.file_list = ScrolledText(file_frame, height=6, wrap="none")
        self.file_list.pack(fill="x", pady=(8, 0))
        self.file_list.configure(state=DISABLED)

        settings = ttk.LabelFrame(outer, text="2. 精度・実行設定", padding=10)
        settings.pack(fill="x", pady=(12, 0))

        row1 = ttk.Frame(settings)
        row1.pack(fill="x")
        ttk.Label(row1, text="モデル", width=12).pack(side="left")
        ttk.Combobox(
            row1,
            textvariable=self.model_var,
            values=list(MODEL_OPTIONS.keys()),
            state="readonly",
            width=38
        ).pack(side="left")
        ttk.Label(
            row1,
            text="※ large-v3 が精度優先",
        ).pack(side="left", padx=(10, 0))

        row2 = ttk.Frame(settings)
        row2.pack(fill="x", pady=(8, 0))
        ttk.Label(row2, text="実行デバイス", width=12).pack(side="left")
        ttk.Combobox(
            row2,
            textvariable=self.device_var,
            values=["自動", "CPU", "NVIDIA GPU"],
            state="readonly",
            width=16
        ).pack(side="left")
        ttk.Label(row2, text="Beam", width=8).pack(side="left", padx=(16, 0))
        ttk.Spinbox(row2, from_=1, to=10, textvariable=self.beam_var, width=5).pack(side="left")
        ttk.Checkbutton(
            row2,
            text="無音区間を除外（VAD）",
            variable=self.vad_var
        ).pack(side="left", padx=(16, 0))

        ttk.Label(
            settings,
            text="固有名詞・専門用語の認識ヒント（任意）"
        ).pack(anchor="w", pady=(10, 2))
        self.prompt_text = ScrolledText(settings, height=4, wrap="word")
        self.prompt_text.pack(fill="x")
        self.prompt_text.insert(
            "1.0",
            "例：舞黒館の惨劇、真歩流、愛理、零度警部、ファイアーエムブレム、生成AI"
        )

        output_frame = ttk.LabelFrame(outer, text="3. 保存先", padding=10)
        output_frame.pack(fill="x", pady=(12, 0))
        out_row = ttk.Frame(output_frame)
        out_row.pack(fill="x")
        ttk.Entry(out_row, textvariable=self.output_var).pack(side="left", fill="x", expand=True)
        ttk.Button(out_row, text="変更", command=self._choose_output).pack(side="left", padx=(8, 0))
        ttk.Button(out_row, text="開く", command=self._open_output).pack(side="left", padx=(8, 0))
        ttk.Label(
            output_frame,
            text="TXT / タイムスタンプTXT / SRT / VTT / JSON を同時に保存します。"
        ).pack(anchor="w", pady=(6, 0))

        exec_frame = ttk.Frame(outer)
        exec_frame.pack(fill="x", pady=(14, 0))

        self.start_button = ttk.Button(
            exec_frame, text="文字起こし開始", command=self._start
        )
        self.start_button.pack(side="left")

        self.cancel_button = ttk.Button(
            exec_frame, text="キャンセル", command=self._cancel, state=DISABLED
        )
        self.cancel_button.pack(side="left", padx=(8, 0))

        self.progress = ttk.Progressbar(exec_frame, orient=HORIZONTAL, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(16, 0))

        ttk.Label(outer, textvariable=self.status_var).pack(anchor="w", pady=(8, 4))

        result_frame = ttk.LabelFrame(outer, text="文字起こし結果（最後に処理したファイル）", padding=8)
        result_frame.pack(fill="both", expand=True)
        self.result_text = ScrolledText(result_frame, wrap="word")
        self.result_text.pack(fill="both", expand=True)

        copy_row = ttk.Frame(outer)
        copy_row.pack(fill="x", pady=(8, 0))
        ttk.Button(copy_row, text="結果をコピー", command=self._copy_result).pack(side="right")

    def _choose_files(self):
        selected = filedialog.askopenfilenames(title="音声・動画ファイルを選択", filetypes=AUDIO_TYPES)
        if not selected:
            return
        self.files = [Path(p) for p in selected]
        if self.files:
            default_out = self.files[0].parent / "transcriptions"
            self.output_var.set(str(default_out))
        self._refresh_file_list()
        self.status_var.set(f"{len(self.files)}件のファイルを選択しました。")

    def _clear_files(self):
        self.files.clear()
        self._refresh_file_list()
        self.status_var.set("音声ファイルを選択してください。")

    def _refresh_file_list(self):
        self.file_list.configure(state=NORMAL)
        self.file_list.delete("1.0", END)
        for p in self.files:
            self.file_list.insert(END, str(p) + "\n")
        self.file_list.configure(state=DISABLED)

    def _choose_output(self):
        folder = filedialog.askdirectory(title="保存先フォルダを選択")
        if folder:
            self.output_var.set(folder)

    def _open_output(self):
        path = Path(self.output_var.get()).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        try:
            os.startfile(path)  # type: ignore[attr-defined]
        except AttributeError:
            import subprocess, sys
            if sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])

    def _copy_result(self):
        text = self.result_text.get("1.0", END).strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("文字起こし結果をクリップボードにコピーしました。")

    def _cuda_runtime_usable(self) -> tuple[bool, str]:
        """Check whether the CUDA runtime DLLs needed by current CTranslate2 are usable."""
        try:
            if ctranslate2.get_cuda_device_count() <= 0:
                return False, "NVIDIA GPUが検出されませんでした。"
        except Exception as exc:
            return False, f"GPU検出に失敗しました: {exc}"

        # Current faster-whisper/CTranslate2 GPU builds typically require CUDA 12 cuBLAS
        # and cuDNN 9. Try loading both before selecting CUDA automatically.
        required = ["cublas64_12.dll", "cudnn64_9.dll"]
        missing = []
        for dll in required:
            try:
                ctypes.WinDLL(dll)
            except Exception:
                missing.append(dll)

        if missing:
            return False, "不足DLL: " + ", ".join(missing)

        return True, "CUDA利用可能"

    def _detect_device(self) -> tuple[str, str]:
        choice = self.device_var.get()
        if choice == "CPU":
            return "cpu", "int8"

        cuda_ok, reason = self._cuda_runtime_usable()

        if choice == "NVIDIA GPU":
            if cuda_ok:
                return "cuda", "float16"
            self.ui_queue.put((
                "status",
                f"GPU実行に必要なCUDAライブラリがないためCPUを使用します（{reason}）"
            ))
            return "cpu", "int8"

        # Auto
        if cuda_ok:
            return "cuda", "float16"

        self.ui_queue.put((
            "status",
            f"GPUは使用できないためCPUで実行します（{reason}）"
        ))
        return "cpu", "int8"

    def _load_model(self, model_id: str, device: str, compute_type: str) -> tuple[WhisperModel, str, str]:
        key = (model_id, device, compute_type)
        if self.loaded_model is not None and self.loaded_model_key == key:
            return self.loaded_model, device, compute_type

        self.ui_queue.put(("status", "モデルを読み込んでいます。初回はダウンロードが発生します…"))

        try:
            model = WhisperModel(
                model_id,
                device=device,
                compute_type=compute_type,
                download_root=str(MODEL_DIR),
            )
            self.loaded_model = model
            self.loaded_model_key = key
            return model, device, compute_type
        except Exception as exc:
            if device == "cuda":
                self.ui_queue.put((
                    "status",
                    f"GPU読み込みに失敗したためCPUへ切り替えます：{exc}"
                ))
                model = WhisperModel(
                    model_id,
                    device="cpu",
                    compute_type="int8",
                    download_root=str(MODEL_DIR),
                )
                self.loaded_model = model
                self.loaded_model_key = (model_id, "cpu", "int8")
                return model, "cpu", "int8"
            raise

    def _start(self):
        if self.worker and self.worker.is_alive():
            return
        if not self.files:
            messagebox.showwarning("入力ファイルなし", "文字起こしする音声ファイルを選択してください。")
            return

        try:
            beam = int(self.beam_var.get())
            if not 1 <= beam <= 10:
                raise ValueError
        except ValueError:
            messagebox.showwarning("設定エラー", "Beam は1〜10の整数にしてください。")
            return

        output_dir = Path(self.output_var.get()).expanduser()
        if not str(output_dir).strip():
            messagebox.showwarning("保存先なし", "保存先フォルダを指定してください。")
            return

        self.cancel_event.clear()
        self.start_button.configure(state=DISABLED)
        self.cancel_button.configure(state=NORMAL)
        self.progress["value"] = 0
        self.result_text.delete("1.0", END)

        args = {
            "files": list(self.files),
            "output_dir": output_dir,
            "model_label": self.model_var.get(),
            "beam": beam,
            "vad": self.vad_var.get(),
            "prompt": self.prompt_text.get("1.0", END).strip(),
        }

        self.worker = threading.Thread(target=self._run_worker, kwargs=args, daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_event.set()
        self.status_var.set("キャンセル要求を受け付けました。現在の区切りで停止します。")
        self.cancel_button.configure(state=DISABLED)

    def _run_worker(
        self,
        files: list[Path],
        output_dir: Path,
        model_label: str,
        beam: int,
        vad: bool,
        prompt: str,
    ):
        try:
            model_id = MODEL_OPTIONS[model_label]
            device, compute_type = self._detect_device()
            model, actual_device, actual_compute = self._load_model(model_id, device, compute_type)

            total_files = len(files)
            for file_index, audio_path in enumerate(files, 1):
                if self.cancel_event.is_set():
                    break

                self.ui_queue.put((
                    "status",
                    f"[{file_index}/{total_files}] {audio_path.name} を解析しています…"
                ))

                is_kotoba = "kotoba-whisper" in model_id
                transcribe_kwargs = {
                    "language": "ja",
                    "task": "transcribe",
                    "beam_size": beam,
                    "vad_filter": vad,
                    "vad_parameters": {
                        "min_silence_duration_ms": 500,
                        "speech_pad_ms": 200,
                    } if vad else None,
                    "initial_prompt": prompt or None,
                    "condition_on_previous_text": not is_kotoba,
                    "word_timestamps": False,
                    "temperature": 0.0,
                    "compression_ratio_threshold": 2.4,
                    "log_prob_threshold": -1.0,
                    "no_speech_threshold": 0.6,
                    "chunk_length": 15 if is_kotoba else 30,
                }

                def transcribe_current(active_model, active_device, active_compute):
                    segments_iter, info_local = active_model.transcribe(
                        str(audio_path), **transcribe_kwargs
                    )
                    duration_local = float(
                        getattr(info_local, "duration", 0.0) or 0.0
                    )

                    collected: list[dict] = []
                    for seg in segments_iter:
                        if self.cancel_event.is_set():
                            break

                        item = {
                            "start": round(float(seg.start), 3),
                            "end": round(float(seg.end), 3),
                            "text": seg.text,
                            "avg_logprob": round(float(seg.avg_logprob), 4),
                            "no_speech_prob": round(float(seg.no_speech_prob), 4),
                        }
                        collected.append(item)

                        local_progress = (
                            min(1.0, seg.end / duration_local)
                            if duration_local > 0 else 0.0
                        )
                        overall = (
                            (file_index - 1) + local_progress
                        ) / total_files * 100
                        self.ui_queue.put(("progress", overall))
                        self.ui_queue.put((
                            "status",
                            f"[{file_index}/{total_files}] {audio_path.name} "
                            f"{human_duration(seg.end)} / "
                            f"{human_duration(duration_local)}"
                        ))

                    return collected, info_local, duration_local

                try:
                    segments, info, duration = transcribe_current(
                        model, actual_device, actual_compute
                    )
                except RuntimeError as exc:
                    msg = str(exc).lower()
                    cuda_problem = (
                        actual_device == "cuda"
                        and (
                            "cublas" in msg
                            or "cudnn" in msg
                            or "cuda" in msg
                            or ".dll" in msg
                        )
                    )
                    if not cuda_problem:
                        raise

                    self.ui_queue.put((
                        "status",
                        "CUDAライブラリ不足を検出しました。"
                        "CPUに自動切り替えして最初から再試行します…"
                    ))

                    # Drop GPU model, then reload on CPU.
                    self.loaded_model = None
                    self.loaded_model_key = None
                    try:
                        import gc
                        gc.collect()
                    except Exception:
                        pass

                    model = WhisperModel(
                        model_id,
                        device="cpu",
                        compute_type="int8",
                        download_root=str(MODEL_DIR),
                    )
                    self.loaded_model = model
                    self.loaded_model_key = (model_id, "cpu", "int8")
                    actual_device = "cpu"
                    actual_compute = "int8"

                    # Reset the file portion of the progress bar before retry.
                    self.ui_queue.put((
                        "progress",
                        (file_index - 1) / total_files * 100
                    ))
                    segments, info, duration = transcribe_current(
                        model, actual_device, actual_compute
                    )

                if self.cancel_event.is_set():
                    break

                language_prob = getattr(info, "language_probability", None)
                metadata = {
                    "model": model_id,
                    "model_label": model_label,
                    "device": actual_device,
                    "compute_type": actual_compute,
                    "language": "ja",
                    "language_probability": (
                        round(float(language_prob), 4)
                        if language_prob is not None else None
                    ),
                    "duration_seconds": round(duration, 3),
                    "beam_size": beam,
                    "vad_filter": vad,
                    "initial_prompt": prompt or None,
                }

                outputs = write_outputs(audio_path, output_dir, segments, metadata)
                plain_text = "".join(s["text"].strip() for s in segments).strip()
                self.ui_queue.put(("result", plain_text))
                self.ui_queue.put((
                    "status",
                    f"[{file_index}/{total_files}] 完了：{audio_path.name} "
                    f"→ {outputs['TXT'].name}"
                ))
                self.ui_queue.put(("progress", file_index / total_files * 100))

            if self.cancel_event.is_set():
                self.ui_queue.put(("finished", "文字起こしをキャンセルしました。"))
            else:
                self.ui_queue.put((
                    "finished",
                    f"完了しました。{total_files}件を {output_dir} に保存しました。"
                ))

        except Exception:
            self.ui_queue.put(("error", traceback.format_exc()))

    def _drain_ui_queue(self):
        try:
            while True:
                event = self.ui_queue.get_nowait()
                kind = event[0]

                if kind == "status":
                    self.status_var.set(event[1])
                elif kind == "progress":
                    self.progress["value"] = float(event[1])
                elif kind == "result":
                    self.result_text.delete("1.0", END)
                    self.result_text.insert("1.0", event[1])
                elif kind == "finished":
                    self.status_var.set(event[1])
                    self.start_button.configure(state=NORMAL)
                    self.cancel_button.configure(state=DISABLED)
                    if not self.cancel_event.is_set():
                        self.progress["value"] = 100
                elif kind == "error":
                    self.start_button.configure(state=NORMAL)
                    self.cancel_button.configure(state=DISABLED)
                    error_text = event[1]
                    self.status_var.set("エラーが発生しました。")
                    messagebox.showerror(
                        "文字起こしエラー",
                        "処理中にエラーが発生しました。\n\n"
                        + error_text[-3500:]
                    )
        except queue.Empty:
            pass
        self.root.after(100, self._drain_ui_queue)


def main():
    root = Tk()
    app = TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
