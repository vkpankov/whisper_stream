import gc
from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.transcribe import (
    get_speech_timestamps,
    collect_chunks,
    VadOptions,
)
from typing import Tuple
import numpy as np
import av
from av.audio.resampler import AudioResampler
import io


class AudioFeaturesFileStream:
    def __init__(
        self,
        file_name: str,
        apply_vad: bool = False,
        vad_parameters: VadOptions = None,
    ) -> None:
        self.av_container = av.open(
            file_name, metadata_errors="ignore", buffer_size=32768 * 1000
        )
        self.av_frames = self.av_container.decode(audio=0)
        self.feature_extractor = FeatureExtractor()
        self.duration = self.av_container.duration / av.time_base
        features_num = int(
            self.duration
            * self.feature_extractor.sampling_rate
            / self.feature_extractor.hop_length
        )
        self.shape = (80, features_num)
        self.apply_vad = apply_vad
        self.vad_parameters = vad_parameters
        self.current_speech_chunks = []

    def av_read_frames(self, start: int, end: int) -> np.ndarray:
        frames_to_read = (
            (end - start)
            * self.feature_extractor.sampling_rate
            // self.feature_extractor.hop_length
        )
        time_offset = int(start * self.feature_extractor.time_per_frame * av.time_base)
        self.av_container.seek(time_offset, any_frame=True)
        resampler = self._initialize_resampler()
        raw_buffer, samples_num = self._read_and_resample_frames(
            frames_to_read, resampler
        )
        self._finalize_resampling(resampler)
        data = self._convert_raw_buffer_to_array(raw_buffer)
        return data

    def _initialize_resampler(self) -> AudioResampler:
        return AudioResampler(
            format="s16",
            layout="mono",
            rate=self.feature_extractor.sampling_rate,
        )

    def _read_and_resample_frames(
        self, frames_to_read: int, resampler: AudioResampler
    ) -> Tuple[io.BytesIO, int]:
        raw_buffer = io.BytesIO()
        samples_num = 0

        for frame in self.av_frames:
            resampled_frame = resampler.resample(frame)[0].to_ndarray()
            raw_buffer.write(resampled_frame)
            samples_num += resampled_frame.shape[-1]

            if samples_num >= frames_to_read * self.feature_extractor.hop_length:
                break

        return raw_buffer, samples_num

    def _finalize_resampling(self, resampler: AudioResampler):
        resampler.resample(None)
        # Comment from faster_whisper.decode_audio(): It appears that some objects related to the resampler are not freed
        # unless the garbage collector is manually run.
        del resampler
        gc.collect()

    def _convert_raw_buffer_to_array(self, raw_buffer: io.BytesIO) -> np.ndarray:
        data = np.frombuffer(raw_buffer.getvalue(), dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
        return data

    def __getitem__(self, slice):
        start = slice[1].start if slice[1].start is not None else 0
        stop = slice[1].stop

        res = self.av_read_frames(start, stop)

        if self.apply_vad:
            speech_chunks = get_speech_timestamps(res, self.vad_parameters)
            res = collect_chunks(res, speech_chunks)
            self.current_speech_chunks.extend(speech_chunks)

        res = self.feature_extractor(res, padding=False)

        return res
