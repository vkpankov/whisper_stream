from typing import Generator
import av
import numpy as np


def ndarray_to_frame(arr: np.ndarray, source_sampling_rate: int) -> np.ndarray:
    source_audio = arr[np.newaxis, :]
    if source_audio.dtype.kind == "f":
        source_audio = (source_audio * 32768.0).astype(np.int16)
    arr_frame = av.AudioFrame.from_ndarray(source_audio, format="s16", layout="mono")
    arr_frame.sample_rate = source_sampling_rate
    return arr_frame


# For unified usage, converting a stream of NumPy arrays to AudioFrames
def ndarray_to_frame_stream(arr_stream: Generator, source_sr: int) -> Generator:
    for arr in arr_stream:
        yield ndarray_to_frame(arr, source_sr)


class StreamSlicer:
    def __init__(self, generator):
        """
        Initialize a StreamSlicer for obtaining sliced data from a stream.
        Args:
            generator (iterable): An iterable providing a stream of numpy arrays.
        """
        self.generator = generator
        self.buffer = None
        self.current_position = 0

    def slice_data(self, start: int, end: int):
        """
        Slice data from the input stream.

        Args:
            start (int): The starting index of the slice.
            end (int): The ending index of the slice.

        Returns:
            np.ndarray: A numpy array containing the sliced data.
        """

        self.result = []
        overlap_data = None
        while len(self.result) <= end - start:
            if self.buffer is not None:
                count_from_buffer = self.current_position - start
                if count_from_buffer > 0:
                    overlap_data = self.buffer[-count_from_buffer:]
                    self.result.extend(overlap_data)

            if len(self.result) < end - start:
                try:
                    chunk = next(self.generator)
                    self.current_position += len(chunk)
                except StopIteration:
                    return np.array(self.result)

                self.result.extend(chunk)
                if overlap_data is not None:
                    self.buffer = np.concatenate([overlap_data, chunk])
                else:
                    self.buffer = chunk
            if len(self.result) >= end - start:
                sliced_data = np.array(self.result[: end - start])
                return sliced_data
