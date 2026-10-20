import json
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gpu_runtime import GPUStatus, check_gpu, configure_dll_search, select_device
from outputs import srt_timestamp, vtt_timestamp, write_outputs
from transcription import Job, MODEL_OPTIONS, Transcriber

GPU = GPUStatus(True, 'GPU ready', frozenset({'float16', 'int8_float16', 'float32'}))
NO_GPU = GPUStatus(False, 'NVIDIA GPU not found')


def segment(text='こんにちは。', start=0, end=1):
    return SimpleNamespace(text=text, start=start, end=end, avg_logprob=-0.1, no_speech_prob=0.01)


class FakeModel:
    def __init__(self, segments=None):
        self.segments = segments or [segment()]
        self.kwargs = None

    def transcribe(self, audio, **kwargs):
        self.kwargs = kwargs
        return iter(self.segments), SimpleNamespace(duration=2, language_probability=1.0)


class DeviceTests(unittest.TestCase):
    def test_auto_gpu_uses_half_precision(self):
        self.assertEqual(select_device('自動', '自動', GPU)[:2], ('cuda', 'float16'))

    def test_cpu_needs_no_gpu(self):
        self.assertEqual(select_device('CPU', '自動', None)[:2], ('cpu', 'int8'))

    def test_auto_missing_gpu_falls_back(self):
        self.assertEqual(select_device('自動', '自動', NO_GPU)[:2], ('cpu', 'int8'))

    def test_forced_gpu_does_not_silently_use_cpu(self):
        with self.assertRaisesRegex(RuntimeError, 'GPUを指定'):
            select_device('NVIDIA GPU', '自動', NO_GPU)

    def test_low_memory_precision(self):
        self.assertEqual(select_device('NVIDIA GPU', 'int8_float16', GPU)[:2], ('cuda', 'int8_float16'))

    def test_unsupported_precision(self):
        with self.assertRaisesRegex(RuntimeError, '対応していません'):
            select_device('NVIDIA GPU', 'float16', GPUStatus(True, '', frozenset({'float32'})))

    def test_auto_precision_on_older_gpu(self):
        self.assertEqual(select_device('自動', '自動', GPUStatus(True, '', frozenset({'float32'})))[1], 'float32')

    def test_linux_does_not_try_windows_dlls(self):
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1,
                                  get_supported_compute_types=lambda *args: {'float16'})
        with patch('gpu_runtime.sys.platform', 'linux'), patch.dict('sys.modules', ctranslate2=backend):
            self.assertTrue(check_gpu().available)
            self.assertEqual(configure_dll_search(), [])

    def test_windows_missing_runtime_is_explained(self):
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1)
        with patch('gpu_runtime.sys.platform', 'win32'), \
                patch('gpu_runtime.configure_dll_search', return_value=[]), \
                patch('gpu_runtime.ctypes.WinDLL', create=True, side_effect=OSError('missing')), \
                patch.dict('os.environ', {'PATH': ''}), patch.dict('sys.modules', ctranslate2=backend):
            status = check_gpu()
            self.assertFalse(status.available)
            self.assertIn('install_gpu.bat', status.detail)
            self.assertIn('cublas64_12.dll', status.missing)
            self.assertIn('見つからない: cublas64_12.dll', status.diagnostics)

    def test_windows_partial_install_lists_missing_dll(self):
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1)

        def load(name):
            if name == 'cudnn_ops64_9.dll':
                raise OSError('missing')
            return name

        with patch('gpu_runtime.sys.platform', 'win32'), \
                patch('gpu_runtime.configure_dll_search', return_value=[Path('nvidia/cudnn/bin')]), \
                patch('gpu_runtime.ctypes.WinDLL', create=True, side_effect=load), \
                patch('gpu_runtime._DLL_HANDLES', []), \
                patch.dict('os.environ', {'PATH': ''}), patch.dict('sys.modules', ctranslate2=backend):
            status = check_gpu()
            self.assertEqual(status.missing, ('cudnn_ops64_9.dll',))
            self.assertIn('cudnn_ops64_9.dll', status.detail)
            self.assertIn('再実行', status.detail)

    def test_windows_present_dll_reports_load_error(self):
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1)
        paging = OSError('paging file')
        paging.winerror = 1455
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'cublasLt64_12.dll').touch()

            def load(name):
                if name.endswith('cublasLt64_12.dll'):
                    raise paging if Path(name).is_absolute() else OSError('missing')
                return name

            with patch('gpu_runtime.sys.platform', 'win32'), \
                    patch('gpu_runtime.configure_dll_search', return_value=[Path(temp)]), \
                    patch('gpu_runtime.ctypes.WinDLL', create=True, side_effect=load), \
                    patch('gpu_runtime._DLL_HANDLES', []), \
                    patch.dict('os.environ', {'PATH': temp}), patch.dict('sys.modules', ctranslate2=backend):
                status = check_gpu()
        self.assertFalse(status.available)
        self.assertEqual(status.missing, ())
        self.assertIn('cublasLt64_12.dll（仮想メモリ（ページングファイル）不足）', status.detail)

    def test_windows_dll_on_path_is_loaded_by_full_path(self):
        backend = SimpleNamespace(get_cuda_device_count=lambda: 1,
                                  get_supported_compute_types=lambda *args: {'float16'})
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp) / 'cublas64_12.dll').touch()
            handles = []

            def load(name):
                if name == 'cublas64_12.dll':
                    raise OSError('bare names skip PATH')
                return name

            with patch('gpu_runtime.sys.platform', 'win32'), \
                    patch('gpu_runtime.configure_dll_search', return_value=[]), \
                    patch('gpu_runtime.ctypes.WinDLL', create=True, side_effect=load), \
                    patch('gpu_runtime._DLL_HANDLES', handles), \
                    patch.dict('os.environ', {'PATH': temp}), patch.dict('sys.modules', ctranslate2=backend):
                self.assertTrue(check_gpu().available)
            self.assertIn(str(Path(temp) / 'cublas64_12.dll'), handles)

    def test_pip_dll_directories_registered_once_and_kept_alive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            targets = [root / 'nvidia' / package / 'bin' for package in ('cublas', 'cudnn')]
            for target in targets:
                target.mkdir(parents=True)
            (root / 'nvidia' / 'cudnn' / 'include').mkdir()
            targets = [target.resolve() for target in targets]
            handles = []
            with patch('gpu_runtime.sys.platform', 'win32'), \
                    patch('gpu_runtime._package_roots', return_value=[root, root]), \
                    patch('gpu_runtime.os.add_dll_directory', create=True, return_value='handle') as add, \
                    patch('gpu_runtime._DLL_HANDLES', handles), patch('gpu_runtime._REGISTERED', set()), \
                    patch.dict('os.environ'):
                self.assertEqual(configure_dll_search(), targets)
                configure_dll_search()
                self.assertEqual([call.args[0] for call in add.call_args_list], [str(t) for t in targets])
                self.assertEqual(handles, ['handle', 'handle'])


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.job = Job((self.root / '音声.wav',), self.root / 'out', next(iter(MODEL_OPTIONS)))
        self.cancel = threading.Event()
        self.events = []

    def run_job(self, engine, job=None):
        engine.run(job or self.job, self.cancel, lambda *event: self.events.append(event))

    def test_gpu_writes_real_device_and_all_formats(self):
        model = FakeModel()
        factory = Mock(return_value=model)
        self.run_job(Transcriber(factory, lambda: GPU))
        factory.assert_called_once()
        self.assertEqual(factory.call_args.kwargs['device'], 'cuda')
        self.assertEqual(model.kwargs['language'], 'ja')
        self.assertIsNone(model.kwargs['initial_prompt'])
        payload = json.loads((self.job.output_dir / '音声.json').read_text(encoding="utf-8"))
        self.assertEqual(payload['device'], 'cuda')
        self.assertEqual(payload['compute_type'], 'float16')
        self.assertEqual(len(list(self.job.output_dir.iterdir())), 5)
        self.assertEqual(self.events[-1][0], 'finished')

    def test_auto_cpu_has_visible_warning(self):
        self.run_job(Transcriber(Mock(return_value=FakeModel()), lambda: NO_GPU))
        self.assertIn('warning', [kind for kind, _ in self.events])
        payload = json.loads((self.job.output_dir / '音声.json').read_text(encoding="utf-8"))
        self.assertEqual(payload['device'], 'cpu')

    def test_strict_gpu_never_constructs_cpu_model(self):
        factory = Mock(side_effect=RuntimeError('CUDA out of memory'))
        job = Job(self.job.files, self.job.output_dir, self.job.model_label, device='NVIDIA GPU')
        self.run_job(Transcriber(factory, lambda: GPU), job)
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.events[-1][0], 'error')
        self.assertFalse(self.job.output_dir.exists())

    def test_auto_gpu_load_failure_retries_on_cpu(self):
        factory = Mock(side_effect=[RuntimeError('CUDA out of memory'), FakeModel()])
        self.run_job(Transcriber(factory, lambda: GPU))
        self.assertEqual([call.kwargs['device'] for call in factory.call_args_list], ['cuda', 'cpu'])
        self.assertEqual(self.events[-1][0], 'finished')

    def test_non_cuda_error_is_not_masked_by_cpu_fallback(self):
        factory = Mock(side_effect=RuntimeError('download failed'))
        self.run_job(Transcriber(factory, lambda: GPU))
        self.assertEqual(factory.call_count, 1)
        self.assertEqual(self.events[-1][0], 'error')

    def test_lazy_gpu_error_clears_partial_result_and_restarts_file(self):
        def failing_segments():
            yield segment('破棄するGPU結果')
            raise RuntimeError('cudnn execution failed')
        gpu_model = FakeModel()
        gpu_model.transcribe = lambda *a, **k: (failing_segments(), SimpleNamespace(duration=2))
        factory = Mock(side_effect=[gpu_model, FakeModel([segment('CPUの結果')])])
        self.run_job(Transcriber(factory, lambda: GPU))
        previews = [data for kind, data in self.events if kind == 'preview']
        self.assertEqual(previews, [(0, ''), (0, '')])
        payload = json.loads((self.job.output_dir / '音声.json').read_text(encoding="utf-8"))
        self.assertEqual(payload['text'], 'CPUの結果')
        self.assertEqual(payload['device'], 'cpu')
        self.assertEqual(self.events[-1][0], 'finished')

    def test_cancellation_does_not_save_partial_file(self):
        class CancellingModel:
            def transcribe(inner, *args, **kwargs):
                def chunks():
                    yield segment()
                    self.cancel.set()
                    yield segment('保存しない')
                return chunks(), SimpleNamespace(duration=2)
        self.run_job(Transcriber(Mock(return_value=CancellingModel()), lambda: GPU))
        self.assertEqual(self.events[-1][0], 'cancelled')
        self.assertFalse(self.job.output_dir.exists())

    def test_cancel_before_model_load(self):
        self.cancel.set()
        factory = Mock()
        self.run_job(Transcriber(factory, lambda: GPU))
        factory.assert_not_called()
        self.assertEqual(self.events[-1][0], 'cancelled')

    def test_model_cache_reused_for_same_device_and_model(self):
        factory = Mock(return_value=FakeModel())
        engine = Transcriber(factory, lambda: GPU)
        self.run_job(engine)
        self.run_job(engine)
        self.assertEqual(factory.call_count, 1)

    def test_kotoba_and_prompt_options(self):
        model = FakeModel()
        label = next(label for label in MODEL_OPTIONS if 'Kotoba' in label)
        job = Job(self.job.files, self.job.output_dir, label, device='CPU', prompt='舞黒館')
        gpu_check = Mock(side_effect=AssertionError('CPU must not probe GPU'))
        self.run_job(Transcriber(Mock(return_value=model), gpu_check), job)
        self.assertFalse(model.kwargs['condition_on_previous_text'])
        self.assertEqual(model.kwargs['chunk_length'], 15)
        self.assertEqual(model.kwargs['initial_prompt'], '舞黒館')
        self.assertEqual(self.events[-1][0], 'finished')


class OutputTests(unittest.TestCase):
    def test_timestamps(self):
        self.assertEqual(srt_timestamp(3661.234), '01:01:01,234')
        self.assertEqual(vtt_timestamp(-1), '00:00:00.000')
        self.assertEqual(srt_timestamp(59.9999), '00:01:00,000')

    def test_unicode_and_subtitle_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = write_outputs(Path('テスト.mp3'), Path(temp),
                                  [{'start': 0, 'end': 1.25, 'text': ' 日本語。'}], {'device': 'cpu'})
            self.assertTrue(paths['TXT'].read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(paths['TXT'].read_text(encoding='utf-8-sig'), '日本語。')
            self.assertIn('00:00:00,000 --> 00:00:01,250', paths['SRT'].read_text(encoding='utf-8-sig'))
            self.assertTrue(paths['VTT'].read_text(encoding="utf-8").startswith('WEBVTT\n'))
            self.assertEqual(json.loads(paths['JSON'].read_text(encoding="utf-8"))['segments'][0]['text'], ' 日本語。')


if __name__ == '__main__':
    unittest.main()
