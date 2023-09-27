from collections import deque
import itertools
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import (
    TranscriptionInfo,
    Segment,
    Word,
    VadOptions,
    get_speech_timestamps,
    collect_chunks,
    restore_speech_timestamps,
)
from backup.streaming_file_reader import decode_audio_stream
from faster_whisper.audio import _resample_frames
import utils
import av

AudioStream = NamedTuple(
    "AudioStream", [("sample_rate", int), ("stream", Iterable[np.ndarray])]
)


class StreamingWhisperModel(WhisperModel):
    def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray, AudioStream],
        language: Optional[str] = None,
        task: str = "transcribe",
        beam_size: int = 5,
        best_of: int = 5,
        patience: float = 1,
        length_penalty: float = 1,
        repetition_penalty: float = 1,
        no_repeat_ngram_size: int = 0,
        temperature: Union[float, List[float], Tuple[float, ...]] = [
            0.0,
            0.2,
            0.4,
            0.6,
            0.8,
            1.0,
        ],
        compression_ratio_threshold: Optional[float] = 2.4,
        log_prob_threshold: Optional[float] = -1.0,
        no_speech_threshold: Optional[float] = 0.6,
        condition_on_previous_text: bool = True,
        prompt_reset_on_temperature: float = 0.5,
        initial_prompt: Optional[Union[str, Iterable[int]]] = None,
        prefix: Optional[str] = None,
        suppress_blank: bool = True,
        suppress_tokens: Optional[List[int]] = [-1],
        without_timestamps: bool = False,
        max_initial_timestamp: float = 1.0,
        word_timestamps: bool = False,
        prepend_punctuations: str = "\"'“¿([{-",
        append_punctuations: str = "\"'.。,，!！?？:：”)]}、",
        vad_filter: bool = False,
        vad_parameters: Optional[Union[dict, VadOptions]] = None,
        max_chunk_size=448000,
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform, or stream of audio chunks (AudioStream object).
          language: The language spoken in the audio. It should be a language code such
            as "en" or "fr". If not set, the language will be detected in the first 30 seconds
            of audio.
          task: Task to execute (transcribe or translate).
          beam_size: Beam size to use for decoding.
          best_of: Number of candidates when sampling with non-zero temperature.
          patience: Beam search patience factor.
          length_penalty: Exponential length penalty constant.
          repetition_penalty: Penalty applied to the score of previously generated tokens
            (set > 1 to penalize).
          no_repeat_ngram_size: Prevent repetitions of ngrams with this size (set 0 to disable).
          temperature: Temperature for sampling. It can be a tuple of temperatures,
            which will be successively used upon failures according to either
            `compression_ratio_threshold` or `log_prob_threshold`.
          compression_ratio_threshold: If the gzip compression ratio is above this value,
            treat as failed.
          log_prob_threshold: If the average log probability over sampled tokens is
            below this value, treat as failed.
          no_speech_threshold: If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
          condition_on_previous_text: If True, the previous output of the model is provided
            as a prompt for the next window; disabling may make the text inconsistent across
            windows, but the model becomes less prone to getting stuck in a failure loop,
            such as repetition looping or timestamps going out of sync.
          prompt_reset_on_temperature: Resets prompt if temperature is above this value.
            Arg has effect only if condition_on_previous_text is True.
          initial_prompt: Optional text string or iterable of token ids to provide as a
            prompt for the first window.
          prefix: Optional text to provide as a prefix for the first window.
          suppress_blank: Suppress blank outputs at the beginning of the sampling.
          suppress_tokens: List of token IDs to suppress. -1 will suppress a default set
            of symbols as defined in the model config.json file.
          without_timestamps: Only sample text tokens.
          max_initial_timestamp: The initial timestamp cannot be later than this.
          word_timestamps: Extract word-level timestamps using the cross-attention pattern
            and dynamic time warping, and include the timestamps for each word in each segment.
          prepend_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the next word
          append_punctuations: If word_timestamps is True, merge these punctuation symbols
            with the previous word
          vad_filter: Enable the voice activity detection (VAD) to filter out parts of the audio
            without speech. This step is using the Silero VAD model
            https://github.com/snakers4/silero-vad.
          vad_parameters: Dictionary of Silero VAD parameters or VadOptions class (see available
            parameters and default values in the class `VadOptions`).
          max_chunk_size: Maximum audio duration that Whisper processes at once (seconds). Longer is better for quality, but requires more memory

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        pos = 0
        promts = deque(maxlen=224)
        last_seek = 0
        last_id = 0
        last_end = 0

        if isinstance(audio, np.ndarray):
            audio = [audio]
        elif isinstance(audio, str):
            container = av.open(audio, metadata_errors="ignore")
            audio = decode_audio_stream(
                container,
                sampling_rate=self.feature_extractor.sampling_rate,
                chunk_size=479830,
            )
        elif isinstance(audio, AudioStream):
            audio = utils.resample_ndarray_stream(
                audio.stream, audio.sample_rate, self.feature_extractor.sampling_rate
            )

        for audio_chunk in audio:
            """if vad_filter:
            voptions = VadOptions()
            speech_chunks = get_speech_timestamps(audio_chunk, voptions)
            audio_chunk = collect_chunks(audio_chunk, speech_chunks)"""

            transcript_segments, info = super().transcribe(
                audio_chunk,
                language="en",
                beam_size=5,
                word_timestamps=True,
                initial_prompt=promts,
            )
            pos += len(audio_chunk)
            if last_seek > 0:
                pos_diff = (
                    len(audio_chunk) - last_seek * self.feature_extractor.hop_length
                )
                pos = pos - pos_diff
                container.seek(int(pos / 16000 * av.time_base))
            else:
                pos_diff = 0

            # transcript_segments = restore_speech_timestamps(transcript_segments, [{"start": pos, "end": pos + len(audio_chunk)}], self.feature_extractor.sampling_rate)
            first_segment_seek = None
            for segment in transcript_segments:
                if first_segment_seek is None:
                    first_seek = segment.seek
                elif segment.seek != first_seek:
                    break

                segment = Segment(
                    id=segment.id + last_id,
                    seek=segment.seek,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                    words=(
                        [Word(**word) for word in segment.words]
                        if word_timestamps
                        else None
                    ),
                )

                promts.extend(segment.tokens)
                yield segment

            last_seek = first_seek
            last_id = segment.id
