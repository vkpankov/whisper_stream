import av
import numpy as np

def resample_ndarray(arr, source_sampling_rate, target_sampling_rate):
    resampler = av.audio.resampler.AudioResampler(format="s16", layout="mono", rate=target_sampling_rate)
    source_audio =arr[np.newaxis,:]
    if source_audio.dtype.kind == 'f':
        source_audio = (source_audio * 32768.0).astype(np.int16)
    arr_frame = av.AudioFrame.from_ndarray(source_audio, format='s16', layout="mono")
    arr_frame.sample_rate = source_sampling_rate
    arr_frame = resampler.resample(arr_frame)
    audio = arr_frame[0].to_ndarray().astype(np.float32) / 32768.0
    return audio[0]

def resample_ndarray_stream(arr_stream, source_sr, target_sr):
    for arr in arr_stream:
        yield resample_ndarray(arr, source_sr, target_sr)