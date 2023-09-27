from typing import Generator
import av
import numpy as np


def resample_ndarray(
    arr: np.ndarray, source_sampling_rate: int, target_sampling_rate: int
) -> np.ndarray:
    resampler = av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=target_sampling_rate
    )

    if arr.dtype.kind == "f":
        arr = (arr * 32768.0).astype(np.int16)

    audio_frame = av.AudioFrame.from_ndarray(
        arr[np.newaxis, :], format="s16", layout="mono"
    )
    audio_frame.sample_rate = source_sampling_rate

    resampled_frame = resampler.resample(audio_frame)
    audio = resampled_frame[0].to_ndarray().astype(np.float32) / 32768.0

    return audio


def resample_ndarray_stream(
    arr_stream: Generator, source_sr: int, target_sr: int
) -> Generator:
    for arr in arr_stream:
        yield resample_ndarray(arr, source_sr, target_sr)


def group_chunks_stream(audio: Generator, min_len: int):
    chunks = []
    total_len = 0

    for audio_chunk in audio:
        chunks.append(audio_chunk)
        total_len += len(audio_chunk)

        if total_len >= min_len:
            yield np.concatenate(chunks)
            chunks = []
            total_len = 0

    if len(chunks) > 0:
        yield np.concatenate(chunks)
