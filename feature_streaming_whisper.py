from collections import deque
import itertools
from typing import BinaryIO, Generator, Iterable, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from faster_whisper import WhisperModel
from faster_whisper.transcribe import *
from backup.streaming_file_reader import decode_audio_stream
from faster_whisper.audio import _resample_frames
import utils
import av
from stream_np import AudioFeaturesFileStream
import utils

AudioStream = NamedTuple("AudioStream", [("sample_rate", int), ("stream", Iterable[np.ndarray])])


class FeatureStreamingWhisperModel(WhisperModel):
  
   def __get_initial_chunk(self, audio):
      initial_chunk = []
      initial_total_len = 0
      for audio_chunk in audio:
          initial_chunk.append(audio_chunk)
          initial_total_len += len(audio_chunk)
          if initial_total_len >= self.feature_extractor.n_samples:
              break
      initial_chunk = np.concatenate(initial_chunk)
      return initial_chunk
   
   def __detect_language(self, language, audio):
      
      if isinstance(audio, Generator):
          initial_chunk = self.__get_initial_chunk(audio)
          lang_segment = self.feature_extractor(initial_chunk[:self.feature_extractor.n_samples], padding=False)
      else:
          initial_chunk = None
          lang_segment = audio[:, : self.feature_extractor.nb_max_frames]
  
      encoder_output = None
      all_language_probs = None
      if language is None:
          if not self.model.is_multilingual:
              language = "en"
              language_probability = 1
          else:
              encoder_output = self.encode(lang_segment)
              # results is a list of tuple[str, float] with language names and
              # probabilities.
              results = self.model.detect_language(encoder_output)[0]
              # Parse language names to strip out markers
              all_language_probs = [(token[2:-2], prob) for (token, prob) in results]
              # Get top language token and probability
              language, language_probability = all_language_probs[0]

              self.logger.info(
                  "Detected language '%s' with probability %.2f",
                  language,
                  language_probability,
              )
      else:
          if not self.model.is_multilingual and language != "en":
              self.logger.warning(
                  "The current model is English-only but the language parameter is set to '%s'; "
                  "using 'en' instead." % language
              )
              language = "en"

          language_probability = 1
      return language, language_probability, all_language_probs, encoder_output, initial_chunk
   
   def audiostream_generate_segments(self, audiostream, tokenizer, options, encoder_output):
        pos = 0
        promts = deque(maxlen=224)
        last_seek = 0
        last_id = 0

        for audio_chunk in audiostream:
            features = self.feature_extractor(audio_chunk)
            transcript_segments = self.generate_segments(features, tokenizer, options, encoder_output)
            transcript_segments = restore_speech_timestamps(transcript_segments, [{"start": pos, "end": pos + len(audio_chunk)}], self.feature_extractor.sampling_rate)
            pos += len(audio_chunk)
            for segment in transcript_segments:
                segment = Segment(
                    id=segment.id + last_id,
                    seek=segment.seek + last_seek,
                    start=segment.start,
                    end=segment.end,
                    text=segment.text,
                    tokens=segment.tokens,
                    temperature=segment.temperature,
                    avg_logprob=segment.avg_logprob,
                    compression_ratio=segment.compression_ratio,
                    no_speech_prob=segment.no_speech_prob,
                    words=segment.words
                )
                promts.extend(segment.tokens)
                yield segment

            last_seek = segment.seek
            last_id = segment.id
            


   def transcribe(
        self,
        audio: Union[str, BinaryIO, np.ndarray],
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
    ) -> Tuple[Iterable[Segment], TranscriptionInfo]:
        """Transcribes an input file.

        Arguments:
          audio: Path to the input file (or a file-like object), or the audio waveform.
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

        Returns:
          A tuple with:

            - a generator over transcribed segments
            - an instance of TranscriptionInfo
        """
        sampling_rate = self.feature_extractor.sampling_rate
        is_audio_stream = isinstance(audio, AudioStream)
        if isinstance(audio, str):
          audio = AudioFeaturesFileStream(audio, vad_filter, vad_parameters)
          duration = audio.duration
        elif is_audio_stream:
          if audio.sample_rate != self.feature_extractor.sampling_rate:
            audio = utils.resample_ndarray_stream(audio.stream, audio.sample_rate, self.feature_extractor.sampling_rate)
          else:
            audio = audio.stream
          duration = 0
        else:
          return super().transcribe(audio) #TODO: all arguments

        duration_after_vad = duration

        self.logger.info(
            "Processing audio with duration %s", format_timestamp(duration)
        )
        

        language, language_probability, all_language_probs, encoder_output, initial_chunk = self.__detect_language(language, audio)

        tokenizer = Tokenizer(
            self.hf_tokenizer,
            self.model.is_multilingual,
            task=task,
            language=language,
        )

        options = TranscriptionOptions(
            beam_size=beam_size,
            best_of=best_of,
            patience=patience,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
            compression_ratio_threshold=compression_ratio_threshold,
            condition_on_previous_text=condition_on_previous_text,
            prompt_reset_on_temperature=prompt_reset_on_temperature,
            temperatures=(
                temperature if isinstance(temperature, (list, tuple)) else [temperature]
            ),
            initial_prompt=initial_prompt,
            prefix=prefix,
            suppress_blank=suppress_blank,
            suppress_tokens=get_suppressed_tokens(tokenizer, suppress_tokens),
            without_timestamps=without_timestamps,
            max_initial_timestamp=max_initial_timestamp,
            word_timestamps=word_timestamps,
            prepend_punctuations=prepend_punctuations,
            append_punctuations=append_punctuations,
        )

        if is_audio_stream:
          audio = itertools.chain([initial_chunk], audio) if initial_chunk is not None else audio
          segments = self.audiostream_generate_segments(audio, tokenizer, options, encoder_output)
        else:
          segments = self.generate_segments(audio, tokenizer, options, encoder_output)

        if not is_audio_stream and audio.current_speech_chunks:
            segments = restore_speech_timestamps(segments, audio.current_speech_chunks, sampling_rate)

        info = TranscriptionInfo(
            language=language,
            language_probability=language_probability,
            duration=duration,
            duration_after_vad=duration_after_vad,
            transcription_options=options,
            vad_options=vad_parameters,
            all_language_probs=all_language_probs,
        )

        return segments, info