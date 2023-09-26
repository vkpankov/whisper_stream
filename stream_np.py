from faster_whisper.feature_extractor import FeatureExtractor
from faster_whisper.transcribe import get_speech_timestamps, collect_chunks
import numpy as np
import av
import io

class AudioFeaturesFileStream():

    def __init__(self, file_name, apply_vad = False, vad_parameters = None) -> None:
        self.av_container = av.open(file_name, metadata_errors = "ignore")
        self.av_frames = self.av_container.decode(audio=0)
        self.feature_extractor = FeatureExtractor()
        self.duration = self.av_container.duration / av.time_base
        features_num = int(self.duration * self.feature_extractor.sampling_rate / self.feature_extractor.hop_length)
        self.shape = (80, features_num)
        self.apply_vad = apply_vad
        self.vad_parameters = vad_parameters
        self.current_speech_chunks = []

    def av_read_from_to(self, a,b ):
        self.av_container.seek(int(a * self.feature_extractor.time_per_frame * av.time_base), any_frame = True)
        self.resampler = av.audio.resampler.AudioResampler(format="s16",layout="mono",
        rate=self.feature_extractor.sampling_rate,
            )
        raw_buffer = io.BytesIO()
        samples_num = 0
        for frame in self.av_frames:
            res_array = self.resampler.resample(frame)[0].to_ndarray()
            raw_buffer.write(res_array)
            samples_num += res_array.shape[-1]
            if samples_num>= (b-a)*self.feature_extractor.hop_length:
                break
        self.resampler.resample(None)
        data = np.frombuffer(raw_buffer.getbuffer(), dtype=res_array.dtype)
        data = data.astype(np.float32) / 32768.0

        return data[:self.feature_extractor.n_samples]
    
    def __getitem__(self, slice):
        res = self.av_read_from_to(slice[1].start if slice[1].start is not None else 0, slice[1].stop)
        if self.apply_vad:
            speech_chunks = get_speech_timestamps(res, self.vad_parameters)
            res = collect_chunks(res, speech_chunks)
            self.current_speech_chunks.extend(speech_chunks)
        res = self.feature_extractor(res, padding = False)
        return res
