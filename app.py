from __future__ import annotations

import ctypes
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, Canvas, StringVar, Tk, filedialog, messagebox, ttk
from tkinter import font as tkfont
from tkinter.scrolledtext import ScrolledText

from gpu_runtime import check_gpu
from transcription import BASE_DIR, MODEL_OPTIONS, Job, Transcriber

APP_TITLE = "日本語ローカル文字起こし"
AUDIO_TYPES = [("音声・動画", "*.wav *.mp3 *.m4a *.flac *.ogg *.aac *.wma *.opus *.mp4 *.mov *.mkv *.webm"),
               ("すべてのファイル", "*.*")]


class TranscriberApp:
    def __init__(self, root: Tk):
        self.root = root
        root.title(APP_TITLE)
        root.geometry(f"{min(1240, root.winfo_screenwidth() - 60)}x{min(850, root.winfo_screenheight() - 100)}")
        root.minsize(980, 620)
        root.configure(background="#f3f6fb")
        root.protocol("WM_DELETE_WINDOW", self._close)
        self.files: list[Path] = []
        self.results: dict[int, str] = {}
        self.outputs: dict[int, dict] = {}
        self.worker = None
        self.busy = False
        self.diagnosing = False
        self.cancel_event = threading.Event()
        self.ui_queue = queue.Queue()
        self.engine = Transcriber()
        self.locked_controls = []
        self.model_var = StringVar(value=next(iter(MODEL_OPTIONS)))
        self.device_var = StringVar(value="自動")
        self.precision_var = StringVar(value="自動")
        self.beam_var = StringVar(value="5")
        self.vad_var = BooleanVar(value=True)
        self.output_var = StringVar(value=str(BASE_DIR / "transcriptions"))
        self.status_var = StringVar(value="ファイルを追加して「文字起こし開始」を押してください。")
        self.device_status = StringVar(value="実行デバイス：待機中")
        self.gpu_status = StringVar(value="GPUを確認しています…")
        self.warning_var = StringVar(value="")
        self.result_title = StringVar(value="文字起こし結果")
        self.count_var = StringVar(value="0件")
        self.progress_var = StringVar(value="0%")
        self._style()
        self._build_ui()
        root.after(80, self._drain_ui_queue)
        self._diagnose_gpu()

    def _style(self):
        families = tkfont.families(self.root)
        family = next((f for f in ("Yu Gothic UI", "Meiryo", "Noto Sans CJK JP") if f in families), tkfont.nametofont("TkDefaultFont").actual("family"))
        self.family = family
        for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont"):
            tkfont.nametofont(name).configure(family=family, size=10)
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=(family, 10), background="#f3f6fb", foreground="#17283e")
        style.configure("Card.TFrame", background="white")
        style.configure("Card.TLabel", background="white")
        style.configure("Muted.TLabel", foreground="#53657d", background="white")
        style.configure("Heading.TLabel", font=(family, 12, "bold"), background="white")
        style.configure("TButton", padding=(12, 7), background="#e6edf7")
        style.map("TButton", background=[("active", "#d6e3f5")])
        style.configure("Primary.TButton", background="#245edb", foreground="white", font=(family, 11, "bold"))
        style.map("Primary.TButton", background=[("disabled", "#a9bddf"), ("active", "#184fc0")],
                  foreground=[("disabled", "#f8fafc")])
        style.configure("TEntry", padding=6, fieldbackground="white")
        style.configure("TCombobox", padding=6, fieldbackground="white")
        style.map("TCombobox", fieldbackground=[("readonly", "white")])
        style.configure("Treeview", rowheight=30, fieldbackground="white", background="white")
        style.configure("Treeview.Heading", font=(family, 10, "bold"), padding=6)
        style.map("Treeview", background=[("selected", "#dce9ff")], foreground=[("selected", "#123a80")])
        style.configure("TNotebook.Tab", padding=(14, 8))
        style.configure("Warning.TLabel", foreground="#895000", background="#fff1cf", padding=8)
        style.configure("TProgressbar", background="#245edb", troughcolor="#e4eaf3", thickness=9)

    def _button(self, parent, text, command, lock=False, **kwargs):
        button = ttk.Button(parent, text=text, command=command, **kwargs)
        if lock:
            self.locked_controls.append((button, "normal"))
        return button

    def _combo(self, parent, variable, values):
        combo = ttk.Combobox(parent, textvariable=variable, values=values, state="readonly")
        combo.pack(fill="x", pady=(4, 12))
        self.locked_controls.append((combo, "readonly"))
        return combo

    def _text(self, parent, **kwargs):
        return ScrolledText(parent, font=(self.family, 12), background="white", foreground="#17283e",
                            relief="flat", borderwidth=0, padx=14, pady=12, **kwargs)

    def _build_ui(self):
        header = self.header = ttk.Frame(self.root, padding=(22, 12))
        header.pack(fill="x")
        ttk.Label(header, text=APP_TITLE, font=(self.family, 20, "bold")).pack(anchor="w")
        self.subtitle = ttk.Label(header, text="音声・動画から日本語の文章へ。音声は外部サービスに送信されません。",
                                  foreground="#53657d")
        self.subtitle.pack(anchor="w", pady=(3, 0))
        body = ttk.Frame(self.root, padding=(18, 0, 18, 12))
        body.pack(fill="both", expand=True)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        # Scrollable settings preserve access at small window sizes / high DPI.
        sidebar = ttk.Frame(body, style="Card.TFrame", width=340)
        sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        sidebar.grid_propagate(False)
        canvas = Canvas(sidebar, background="white", highlightthickness=0, width=316)
        scroll = ttk.Scrollbar(sidebar, orient="vertical", command=canvas.yview)
        scroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.configure(yscrollcommand=scroll.set)
        settings = ttk.Frame(canvas, style="Card.TFrame", padding=16)
        window = canvas.create_window((0, 0), window=settings, anchor="nw")
        settings.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(window, width=e.width))
        def wheel(event):
            widget = event.widget
            while widget is not None:
                if widget == sidebar:
                    canvas.yview_scroll(-1 if event.delta > 0 or event.num == 4 else 1, "units")
                    return "break"
                widget = getattr(widget, "master", None)
        self.root.bind("<MouseWheel>", wheel, add="+")
        self.root.bind("<Button-4>", wheel, add="+")
        self.root.bind("<Button-5>", wheel, add="+")
        ttk.Label(settings, text="文字起こしの設定", style="Heading.TLabel").pack(anchor="w", pady=(0, 16))
        ttk.Label(settings, text="モデル", style="Card.TLabel").pack(anchor="w")
        self._combo(settings, self.model_var, list(MODEL_OPTIONS))
        ttk.Label(settings, text="実行デバイス", style="Card.TLabel").pack(anchor="w")
        self._combo(settings, self.device_var, ["自動", "NVIDIA GPU", "CPU"])
        ttk.Label(settings, text="GPUの計算精度", style="Card.TLabel").pack(anchor="w")
        self._combo(settings, self.precision_var, ["自動", "float16", "int8_float16", "float32"])
        ttk.Label(settings, text="メモリ不足の場合は int8_float16 をお試しください。",
                  style="Muted.TLabel", wraplength=280).pack(anchor="w")
        ttk.Separator(settings).pack(fill="x", pady=14)
        ttk.Label(settings, text="GPUの状態", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(settings, textvariable=self.gpu_status, style="Muted.TLabel", wraplength=276).pack(anchor="w", pady=8)
        self.gpu_button = self._button(settings, "GPUを再確認", self._diagnose_gpu, lock=True)
        self.gpu_button.pack(fill="x")
        self._button(settings, "GPUセットアップ手順", self._gpu_help).pack(fill="x", pady=(6, 0))
        ttk.Separator(settings).pack(fill="x", pady=14)
        ttk.Label(settings, text="認識ヒント（任意）", style="Heading.TLabel").pack(anchor="w")
        ttk.Label(settings, text="人名・作品名・専門用語などを入力", style="Muted.TLabel").pack(anchor="w", pady=(4, 6))
        self.prompt_text = self._text(settings, height=3, wrap="word")
        self.prompt_text.configure(font=(self.family, 10), borderwidth=1, relief="solid", padx=6, pady=6)
        self.prompt_text.pack(fill="x")
        self.locked_controls.append((self.prompt_text, "normal"))
        tuning = ttk.Frame(settings, style="Card.TFrame")
        tuning.pack(fill="x", pady=(12, 0))
        ttk.Label(tuning, text="Beam（探索数）", style="Card.TLabel").pack(side="left")
        beam = ttk.Spinbox(tuning, from_=1, to=10, textvariable=self.beam_var, width=4)
        beam.pack(side="right")
        self.locked_controls.append((beam, "normal"))
        vad = ttk.Checkbutton(settings, text="無音区間を除外する（VAD）", variable=self.vad_var)
        vad.pack(anchor="w", pady=10)
        self.locked_controls.append((vad, "normal"))
        ttk.Separator(settings).pack(fill="x", pady=(4, 14))
        ttk.Label(settings, text="保存先", style="Heading.TLabel").pack(anchor="w")
        output_entry = ttk.Entry(settings, textvariable=self.output_var)
        output_entry.pack(fill="x", pady=8)
        self.locked_controls.append((output_entry, "normal"))
        out_buttons = ttk.Frame(settings, style="Card.TFrame")
        out_buttons.pack(fill="x")
        self._button(out_buttons, "変更", self._choose_output, lock=True).pack(side="left")
        self._button(out_buttons, "フォルダを開く", self._open_output).pack(side="right")
        ttk.Label(settings, text="TXT・時刻付きTXT・SRT・VTT・JSONを保存",
                  style="Muted.TLabel", wraplength=276).pack(anchor="w", pady=(8, 0))

        main = ttk.Frame(body)
        main.grid(row=0, column=1, sticky="nsew")
        main.columnconfigure(0, weight=1)
        main.rowconfigure(2, weight=1)
        file_card = ttk.Frame(main, style="Card.TFrame", padding=14)
        file_card.grid(row=0, column=0, sticky="ew")
        row = ttk.Frame(file_card, style="Card.TFrame")
        row.pack(fill="x", pady=(0, 10))
        ttk.Label(row, text="入力ファイル", style="Heading.TLabel").pack(side="left")
        ttk.Label(row, textvariable=self.count_var, style="Muted.TLabel").pack(side="left", padx=10)
        self._button(row, "追加", self._choose_files, lock=True).pack(side="right")
        self._button(row, "削除", self._remove_files, lock=True).pack(side="right", padx=6)
        tree_frame = ttk.Frame(file_card, style="Card.TFrame")
        tree_frame.pack(fill="x")
        self.file_tree = ttk.Treeview(tree_frame, columns=("name", "state"), show="headings", height=4)
        self.file_tree.heading("name", text="ファイル名（選択すると結果を表示）")
        self.file_tree.heading("state", text="状態")
        self.file_tree.column("name", width=350, minwidth=120)
        self.file_tree.column("state", width=90, minwidth=90, stretch=False)
        self.file_tree.pack(side="left", fill="x", expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, command=self.file_tree.yview)
        tree_scroll.pack(side="right", fill="y")
        self.file_tree.configure(yscrollcommand=tree_scroll.set)
        self.file_tree.bind("<<TreeviewSelect>>", self._show_selected)
        self.warning_label = ttk.Label(main, textvariable=self.warning_var, style="Warning.TLabel", wraplength=600)

        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=2, column=0, sticky="nsew", pady=(12, 0))
        results = ttk.Frame(self.notebook, style="Card.TFrame", padding=8)
        self.notebook.add(results, text="文字起こし結果")
        result_bar = ttk.Frame(results, style="Card.TFrame")
        result_bar.pack(fill="x", padx=6, pady=(4, 6))
        ttk.Label(result_bar, textvariable=self.result_title, style="Heading.TLabel", wraplength=430).pack(side="left")
        self._button(result_bar, "コピー", self._copy_result).pack(side="right")
        self.result_text = self._text(results, wrap="word", height=8, state="disabled")
        self.result_text.pack(fill="both", expand=True)
        log_frame = ttk.Frame(self.notebook, style="Card.TFrame")
        self.notebook.add(log_frame, text="処理ログ")
        self.log_text = self._text(log_frame, wrap="word", height=8, state="disabled")
        self.log_text.pack(fill="both", expand=True)
        footer = ttk.Frame(self.root, padding=(20, 12))
        footer.pack(side="bottom", fill="x", before=body)
        ttk.Label(footer, textvariable=self.device_status, font=(self.family, 10, "bold")).pack(anchor="w")
        self.status_label = ttk.Label(footer, textvariable=self.status_var, wraplength=920)
        self.status_label.pack(anchor="w", pady=(4, 10))
        action = ttk.Frame(footer)
        action.pack(fill="x")
        self.start_button = self._button(action, "文字起こし開始", self._start, style="Primary.TButton")
        self.start_button.pack(side="right")
        self.cancel_button = self._button(action, "中止", self._cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=8)
        self.progress = ttk.Progressbar(action, mode="determinate")
        self.progress.pack(side="left", fill="x", expand=True, padx=(0, 8))
        ttk.Label(action, textvariable=self.progress_var, width=5).pack(side="left", padx=(0, 10))
        self.root.bind("<Configure>", self._resize, add="+")

    def _resize(self, event):
        if event.widget is self.root:
            if event.height < 700:
                self.subtitle.pack_forget()
                self.header.configure(padding=(22, 8))
            else:
                self.subtitle.pack(anchor="w", pady=(3, 0))
                self.header.configure(padding=(22, 12))
            self.file_tree.configure(height=2 if event.height < 740 else 4)
            self.status_label.configure(wraplength=max(400, event.width - 40))
            self.warning_label.configure(wraplength=max(300, event.width - 420))

    def _selected(self):
        selection = self.file_tree.selection()
        return int(selection[0]) if selection else None

    def _choose_files(self):
        selected = filedialog.askopenfilenames(title="音声・動画を追加", filetypes=AUDIO_TYPES)
        if not selected:
            return
        was_empty = not self.files
        for path in map(Path, selected):
            if path not in self.files:
                self.files.append(path)
                self.file_tree.insert("", "end", iid=str(len(self.files) - 1), values=(path.name, "待機"))
        if was_empty and self.files:
            self.output_var.set(str(self.files[0].parent / "transcriptions"))
        self.count_var.set(f"{len(self.files)}件")
        if self.files and not self.file_tree.selection():
            self.file_tree.selection_set("0")
        self.status_var.set(f"{len(self.files)}件のファイルを選択しました。")

    def _remove_files(self):
        selected = {int(i) for i in self.file_tree.selection()}
        if not selected:
            return
        remaining = [i for i in range(len(self.files)) if i not in selected]
        states = {i: self.file_tree.set(str(i), "state") for i in remaining}
        self.files = [self.files[i] for i in remaining]
        self.results = {n: self.results[i] for n, i in enumerate(remaining) if i in self.results}
        self.outputs = {n: self.outputs[i] for n, i in enumerate(remaining) if i in self.outputs}
        self.file_tree.delete(*self.file_tree.get_children())
        for n, old in enumerate(remaining):
            self.file_tree.insert("", "end", iid=str(n), values=(self.files[n].name, states[old]))
        self.count_var.set(f"{len(self.files)}件")
        if self.files:
            self.file_tree.selection_set("0")
        self._show_selected()

    def _show_selected(self, event=None):
        index = self._selected()
        self.result_title.set(self.files[index].name if index is not None else "文字起こし結果")
        text = self.results.get(index, "")
        self._replace_text(self.result_text, text or "ここに文字起こし結果が表示されます。\n処理中の文章も順次確認できます。")

    @staticmethod
    def _replace_text(widget, text):
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _log(self, text):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _choose_output(self):
        folder = filedialog.askdirectory(title="保存先フォルダを選択")
        if folder:
            self.output_var.set(folder)

    def _open_output(self):
        try:
            if not self.output_var.get().strip():
                raise ValueError("保存先フォルダを指定してください。")
            path = Path(self.output_var.get()).expanduser()
            path.mkdir(parents=True, exist_ok=True)
            if sys.platform == "win32":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("保存先を開けません", str(exc))

    def _copy_result(self):
        text = self.results.get(self._selected(), "")
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.status_var.set("結果をコピーしました。")

    def _diagnose_gpu(self):
        if self.diagnosing or self.busy:
            return
        self.diagnosing = True
        self.gpu_button.configure(state="disabled")
        self.gpu_status.set("GPUを確認しています…")
        def run():
            status = check_gpu()
            self.ui_queue.put(("gpu", status))
        threading.Thread(target=run, daemon=True).start()

    def _gpu_help(self):
        messagebox.showinfo("GPUセットアップ", "NVIDIA GPU向けの設定です。\n\n"
                            "1. アプリを閉じます。\n2. install_gpu.bat をダブルクリックします。\n"
                            "3. 完了後 run.bat で再起動します。\n4. 「GPUを再確認」で状態を確認します。\n\n"
                            "CUDA 12 / cuDNN 9のライブラリを仮想環境に導入します。\n"
                            "NVIDIAドライバーは別途必要です。AMD・Intel GPUには対応していません。\n"
                            "GPU指定時はCPUへの自動切り替えを行いません。")

    def _set_busy(self, busy):
        self.busy = busy
        for widget, state in self.locked_controls:
            widget.configure(state="disabled" if busy else state)
        if self.diagnosing:
            self.gpu_button.configure(state="disabled")
        self.start_button.configure(state="disabled" if busy else "normal")
        self.cancel_button.configure(state="normal" if busy else "disabled")

    def _start(self):
        if self.busy or (self.worker and self.worker.is_alive()):
            return
        try:
            if not self.files:
                raise ValueError("音声・動画ファイルを追加してください。")
            for path in self.files:
                if not path.is_file():
                    raise ValueError(f"ファイルが見つかりません: {path}")
            if not self.output_var.get().strip():
                raise ValueError("保存先フォルダを指定してください。")
            beam = int(self.beam_var.get())
            if not 1 <= beam <= 10:
                raise ValueError("Beamは1〜10の整数で指定してください。")
            stems = [path.stem.casefold() for path in self.files]
            if len(stems) != len(set(stems)):
                raise ValueError("同じ名前のファイルがあります。保存結果が重ならないよう、入力ファイル名を変えてください。")
            job = Job(tuple(self.files), Path(self.output_var.get().strip()).expanduser(),
                      self.model_var.get(), self.device_var.get(), self.precision_var.get(),
                      beam, self.vad_var.get(), self.prompt_text.get("1.0", "end").strip())
            # Validate the directory before downloading a model.
            job.output_dir.mkdir(parents=True, exist_ok=True)
            targets = [job.output_dir / (p.stem + suffix) for p in job.files
                       for suffix in (".txt", "_timestamps.txt", ".srt", ".vtt", ".json")]
            names = [p.name.casefold() for p in targets]
            if len(names) != len(set(names)):
                raise ValueError("出力名が重なるファイルがあります。入力ファイル名を変えてください。")
            if any(p.exists() for p in targets) and not messagebox.askyesno("出力の上書き", "保存先に同名の結果があります。上書きして開始しますか？"):
                return
        except (ValueError, OSError) as exc:
            messagebox.showwarning("入力・設定を確認", str(exc))
            return
        self.cancel_event.clear()
        self.results.clear()
        self.outputs.clear()
        self.warning_var.set("")
        self.warning_label.grid_remove()
        self.progress.configure(mode="determinate", value=0)
        self.progress_var.set("0%")
        self._replace_text(self.log_text, "")
        self._set_busy(True)
        self.notebook.select(0)
        for i in range(len(self.files)):
            self.file_tree.set(str(i), "state", "待機")
        self._show_selected()
        self.worker = threading.Thread(target=self.engine.run,
                                       args=(job, self.cancel_event, lambda kind, data: self.ui_queue.put((kind, data))),
                                       daemon=True)
        self.worker.start()

    def _cancel(self):
        self.cancel_event.set()
        self.status_var.set("中止を待っています。モデル準備・解析中は区切りまで時間がかかる場合があります。")
        self.cancel_button.configure(state="disabled")

    def _close(self):
        if self.busy and not messagebox.askyesno("アプリを終了", "文字起こし中です。途中のファイルは保存されません。終了しますか？"):
            return
        self.cancel_event.set()
        self.root.destroy()

    def _drain_ui_queue(self):
        # Bounded work per tick keeps Tk responsive on large batches.
        for _ in range(150):
            try:
                kind, data = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "gpu":
                self.diagnosing = False
                self.gpu_status.set(data.detail)
                self.gpu_button.configure(state="disabled" if self.busy else "normal")
                self._log("GPU診断: " + data.detail + ("\n" + data.diagnostics if data.diagnostics else ""))
            elif kind in ("phase", "status"):
                if not self.cancel_event.is_set():
                    self.status_var.set(data)
                if kind == "phase":
                    self.progress.configure(mode="indeterminate")
                    self.progress.start(12)
                    self.progress_var.set("準備中")
                    self._log(data)
            elif kind == "device":
                device, compute, detail = data
                self.device_status.set(f"実行デバイス：{'NVIDIA GPU' if device == 'cuda' else 'CPU'} · {compute}")
                self._log(self.device_status.get() + "\n" + detail)
            elif kind == "warning":
                self.warning_var.set("GPUを使用できないためCPUで処理します。理由は「処理ログ」で確認できます。")
                self.warning_label.grid(row=1, column=0, sticky="ew", pady=(10, 0))
                self._log(data)
            elif kind == "progress":
                self.progress.stop()
                self.progress.configure(mode="determinate", value=data)
                self.progress_var.set(f"{data:.0f}%")
            elif kind == "file":
                index, state = data
                self.file_tree.set(str(index), "state", state)
                if state == "処理中":
                    self.file_tree.selection_set(str(index))
                    self.file_tree.see(str(index))
            elif kind == "preview":
                index, text = data
                self.results[index] = text
                if index == self._selected():
                    self._show_selected()
            elif kind == "segment":
                index, text = data
                self.results[index] = self.results.get(index, "") + text
                if index == self._selected():
                    if self.results[index] == text:
                        self._replace_text(self.result_text, "")
                    self.result_text.configure(state="normal")
                    self.result_text.insert("end", text)
                    self.result_text.see("end")
                    self.result_text.configure(state="disabled")
            elif kind == "saved":
                index, outputs = data
                self.outputs[index] = outputs
            elif kind == "log":
                self._log(data)
            elif kind in ("finished", "cancelled", "error"):
                self.progress.stop()
                self.progress.configure(mode="determinate")
                self._set_busy(False)
                self.status_var.set(data if kind != "error" else "エラーが発生しました。「処理ログ」を確認してください。")
                self._log(data)
                for item in self.file_tree.get_children():
                    state = self.file_tree.set(item, "state")
                    if state == "処理中":
                        self.file_tree.set(item, "state", "エラー" if kind == "error" else "中止・未保存")
                    elif state == "待機" and kind != "finished":
                        self.file_tree.set(item, "state", "未処理")
                if kind == "finished":
                    self.progress.configure(value=100)
                    self.progress_var.set("100%")
                elif kind == "error":
                    messagebox.showerror("文字起こしエラー", data)
        self.root.after(80, self._drain_ui_queue)


def main():
    if sys.platform == "win32":
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = Tk()
    TranscriberApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
