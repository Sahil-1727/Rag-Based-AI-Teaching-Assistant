# RAG Based AI Teaching Assistant

An AI teaching assistant that answers questions about a video course (Sigma Web Development) and tells you **which video, and which timestamps**, to watch.

It uses **Retrieval-Augmented Generation (RAG)**: instead of letting a language model guess, the app first finds the most relevant parts of the course transcripts, then asks the model to answer using only those parts.

## How it works

```
Videos -> Audio -> Transcript chunks (JSON) -> Embeddings -> ChromaDB
                                                                |
Question -> Embedding -> Top 5 similar chunks <-----------------+
                              |
                              v
                    Prompt + chunks -> LLM -> Answer with video number and timestamps
```

1. **`vidToMp3.py`** converts the course videos to audio files.
2. **`mp3toJson.py`** turns each audio file into a JSON file of transcript chunks, each with a start time, end time and text.
3. **`preprocess_json.py`** creates an embedding for every chunk using `bge-m3` (through Ollama) and stores the chunks, embeddings and metadata (title, video number, timestamps) in a **ChromaDB** vector database.
4. **`process_incoming.py`** takes a question, embeds it with the same model, asks ChromaDB for the 5 closest chunks, and sends them to an LLM. The LLM answers only from those chunks and points you to the right video.

## Tech stack

- Python
- [Ollama](https://ollama.com): runs `bge-m3` (embeddings) and `llama3.2` (answers) locally
- [ChromaDB](https://www.trychroma.com): vector database (cosine similarity)
- Optional: OpenAI API (`gpt-4o`) as an alternative to `llama3.2`
- pandas, requests

## Setup

1. Clone the repo and install the dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Install Ollama, then pull the two models:
   ```
   ollama pull bge-m3
   ollama pull llama3.2
   ```
3. (Optional) To use OpenAI instead of `llama3.2`, copy `config.example.py` to `config.py` and put your API key in it. `config.py` is in `.gitignore`, so it is never committed.

## Usage

Run the steps in order:

```
python vidToMp3.py          # videos  -> audio
python mp3toJson.py         # audio   -> transcript chunks (jsons/)
python preprocess_json.py   # chunks  -> embeddings -> ChromaDB (chroma_db/)
python process_incoming.py  # ask questions
```

Example:

```
Ask a Question: how do I make a table in HTML?
```

The assistant replies with the answer and the video number, title and timestamps where it is taught.

If the question isn't covered by the course (for example something unrelated), the prompt tells the model to say so instead of making up an answer.

## Notes

- The course videos, audio, transcripts and the generated `chroma_db/` folder are not included in this repo. Run the pipeline on your own course videos to build them.
- Retrieval started as a pandas + `cosine_similarity` search over a saved file, and was moved to ChromaDB so that search is done by a vector database instead of scanning every row in memory.

## Ideas for next steps

- Better chunking of transcripts
- A small set of test questions to measure retrieval quality
- Reranking the retrieved chunks
- An agent layer (LangGraph) that decides when to search
