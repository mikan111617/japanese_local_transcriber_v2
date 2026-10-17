"""Run explicitly on a desktop or with xvfb-run; no model download required."""
import tempfile
import time
import unittest
from pathlib import Path
from tkinter import Tk
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import TranscriberApp
from gpu_runtime import GPUStatus
from transcription import Transcriber

GPU = GPUStatus(True, 'NVIDIA GPU 1台を検出。CUDA利用可能。', frozenset({'float16'}))


class GUITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.paths = [Path(self.temp.name) / name for name in ('会話.wav', 'インタビュー.mp3')]
        for path in self.paths:
            path.touch()
        self.patch_gpu = patch('app.check_gpu', return_value=GPU)
        self.patch_gpu.start()
        self.addCleanup(self.patch_gpu.stop)
        try:
            self.root = Tk()
        except Exception as exc:
            self.skipTest(f'Desktop/Xvfb unavailable: {exc}')
        self.addCleanup(self.root.destroy)
        self.app = TranscriberApp(self.root)
        self.root.update()
        self.pump(lambda: not self.app.diagnosing)

    def pump(self, done):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.root.update()
            if done():
                return
            time.sleep(0.01)
        self.fail('UI event processing timed out')

    def add_files(self):
        with patch('app.filedialog.askopenfilenames', return_value=[str(p) for p in self.paths]):
            self.app._choose_files()
        self.root.update()

    def test_layout_at_small_and_large_sizes(self):
        for geometry in ('1240x850', '980x620'):
            self.root.geometry(geometry)
            self.root.update()
            self.assertGreater(self.app.result_text.winfo_height(), 100)
            self.assertLessEqual(self.app.start_button.winfo_rooty() + self.app.start_button.winfo_height(),
                                 self.root.winfo_rooty() + self.root.winfo_height())
            self.assertGreater(self.app.file_tree.winfo_width(), 200)

    def test_batch_live_results_copy_and_selection(self):
        self.add_files()
        info = SimpleNamespace(duration=2, language_probability=1.0)
        def transcribe(*args, **kwargs):
            seg = SimpleNamespace(start=0, end=2, text='テストの文章です。', avg_logprob=-0.1, no_speech_prob=0.0)
            return iter([seg]), info
        model = SimpleNamespace(transcribe=transcribe)
        self.app.engine = Transcriber(Mock(return_value=model), lambda: GPU)
        self.app._start()
        self.assertTrue(self.app.busy)
        self.assertEqual(str(self.app.start_button['state']), 'disabled')
        self.pump(lambda: not self.app.busy)
        self.assertIn('NVIDIA GPU', self.app.device_status.get())
        self.assertEqual(len(self.app.outputs), 2)
        self.assertEqual(self.app.file_tree.set('0', 'state'), '完了')
        self.assertIn('テストの文章です。', self.app.result_text.get('1.0', 'end'))
        self.app.file_tree.selection_set('0')
        self.root.update()
        self.assertEqual(self.app.result_title.get(), '会話.wav')
        self.app._copy_result()
        self.assertEqual(self.root.clipboard_get(), 'テストの文章です。')
        self.assertEqual(self.app.progress_var.get(), '100%')
        self.app._remove_files()
        self.root.update()
        self.assertEqual(self.app.files, [self.paths[1]])
        self.assertEqual(self.app.results[0], 'テストの文章です。')
        self.assertEqual(self.app.file_tree.set('0', 'state'), '完了')

    def test_empty_output_is_rejected(self):
        self.add_files()
        self.app.output_var.set('   ')
        with patch('app.messagebox.showwarning') as warning:
            self.app._start()
        warning.assert_called_once()
        self.assertFalse(self.app.busy)

    def test_cross_format_filename_collision_is_rejected(self):
        paths = [Path(self.temp.name) / name for name in ('音声.wav', '音声_timestamps.mp3')]
        for path in paths:
            path.touch()
        with patch('app.filedialog.askopenfilenames', return_value=[str(p) for p in paths]):
            self.app._choose_files()
        with patch('app.messagebox.showwarning') as warning:
            self.app._start()
        self.assertIn('出力名が重なる', warning.call_args.args[1])
        self.assertFalse(self.app.busy)


if __name__ == '__main__':
    unittest.main()
