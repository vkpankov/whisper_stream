from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.transcribe import get_speech_timestamps, collect_chunks, VadOptions
from faster_whisper.audio import _resample_frames, _ignore_invalid_frames, _group_frames
import numpy as np
import av
import gc
from utils import StreamSlicer


class FramesStream:
    def __init__(
        self,
        av_frames,
        chunk_size: int = 480000,
        sampling_rate: int = 16000,
        apply_vad: bool = False,
        vad_parameters: VadOptions = None,
    ) -> None:
        self.av_frames = av_frames
        self.apply_vad = apply_vad
        if apply_vad and vad_parameters is None:
            vad_parameters = VadOptions()
        self.vad_parameters = vad_parameters
        self.sampling_rate = sampling_rate
        self.current_speech_chunks = []
        self.av_frames = _ignore_invalid_frames(self.av_frames)
        self.av_frames = _group_frames(
            self.av_frames,
            chunk_size,
        )
        self.resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=sampling_rate,
        )
        self.av_frames = _resample_frames(self.av_frames, self.resampler)
        self.stream = self._apply_resample_vad(self.av_frames, chunk_size)
        self.stream_slicer = StreamSlicer(self.stream)

    def __del__(self):
        # Comment from faster_whisper.decode_audio: It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        gc.collect()

    def _apply_resample_vad(self, input_stream, output_chunk_size: int):
        """
        Generate a new stream with a different chunk size from an input stream.

        Args:
            input_stream (iterable): Input stream of NumPy arrays
            output_chunk_size (int): Desired chunk size of the output stream.

        Yields:
            np.ndarray: Chunks of the output stream with the specified chunk size.
        """
        current_chunk = np.array([], dtype=np.float32)
        global_pos = 0
        for frame in input_stream:
            if not isinstance(frame, np.ndarray):
                frame = frame.to_ndarray()[0]
            fmax = np.abs(frame).max()
            if fmax == 0:
                global_pos += len(frame)
                continue
            frame = (frame / float(fmax)).astype(np.float32)

            if self.apply_vad:
                speech_chunks = get_speech_timestamps(frame, self.vad_parameters)

                origin_len = len(frame)
                frame = collect_chunks(frame, speech_chunks)
                for i in range(len(speech_chunks)):
                    speech_chunks[i]["start"] = global_pos + speech_chunks[i]["start"]
                    speech_chunks[i]["end"] = global_pos + speech_chunks[i]["end"]

                global_pos += origin_len
                self.current_speech_chunks.extend(speech_chunks)

            current_chunk = np.concatenate((current_chunk, frame))

            while len(current_chunk) >= output_chunk_size:
                yield current_chunk[:output_chunk_size]
                current_chunk = current_chunk[output_chunk_size:]

        if len(current_chunk) > 0:
            yield current_chunk

    def __getitem__(self, slice):
        start = slice.start if slice.start is not None else 0
        res = self.stream_slicer.slice_data(start, slice.stop)
        return res
