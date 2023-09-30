## Notes:
 - Could not avoid copying most of the generate_segments function as the original version was designed for features, and it presented two main challenges: 1) it was not designed for streaming without a known audio length, and 2) when fed short segments (e.g., 3000 milliseconds), it could run a few times with shorter parts of the segment. (However, it can be refactored) 
 - Compared to the non-streaming method, there are inevitable differences such as chunk-wise feature extraction, resampling, VAD: potential reasons why the results may not be identical.
 - vad_duration always None (it is not possible to estimate it in advance)
 - Also tried https://github.com/ufal/whisper_streaming , but it's very slow and gives worse metrics (due to hallucinations) on two tested audios


## Possible algorithms for reducing the difference in the speed of pronunciation of source / translated text
 - Replace words in the sentence with longer or shorter synonyms, and retrieve contextually similar words using a model like Word2Vec.
 - Text summarization with control of output length
    - Tried https://huggingface.co/facebook/bart-large-cnn . Very low quality results (hallucinations) in case of text expansion
 - LLM: 1) finetune open-source for summarization task 2) ChatGPT API
