# ReviewAnalyticsDHA
Literature review on data analytics for digital health applications.


# Usage

- pip install -r requirements.txt
- run main_dense.py with the following arguments (see argument description below):
  
python main_dense.py --papers-dir {PAPERS_DIR} --output-dir {OUTPUT_DIR} --questions-file {QUESTIONS_FILE} --model {LLM} --top-k {K} --chunk-tokens {N_TOKENS_PER_CHUNK} --api-key-file {API_KEY_FILEPATH} --dense-model {SENTENCE_EMBEDDER} --dense-batch-size {BATCH_SIZE}


- PAPERS_DIR: directory containing the papers in pdf format (e.g. ./papers/).
- OUTPUT_DIR: directory to output answers to.
- QUESTIONS_FILE: JSON file containing the questions. Can be either e.g. utils/questions_free_text.json or utils/questions_labeled.json. Will automatically choose between free-text or structured extraction, dependent on whether labels are defined within the questions file.
- LLM: the OpenRouter LLM to use. See the OpenRouter website for which LLMs are available. Defaults to DeepSeek.
- K: number of most appropriate text chunks to put into LLM context. Defaults to 3.
- N_TOKENS_PER_CHUNK: number of tokens to partition articles into. Defaults to 800.
- API_KEY_FILEPATH: .txt file containing an OpenRouter API key. Note that even when using free tier LLMs, requests are heavily throttled if you're not putting at least 10$ worth of credits on your account.
- SENTENCE-EMBEDDER: sentence embedding model to use from the transformers library. Defaults to kamalkraj/BioSimCSE-BioLinkBERT-BASE. 
- BATCH_SIZE: batch size for sentence embedder. Defaults to 8.
