import requests
import os
import json
import pandas as pd
import joblib
import chromadb 


def create_embedding(text_list):
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })

    response_data = r.json()

    # Check if the API returned an error
    if "error" in response_data:
        raise ValueError(f"Ollama API Error: {response_data['error']}")

    # Use the plural "embeddings" key
    return response_data["embeddings"]


jsons = os.listdir("jsons")  # List all the jsons
my_dicts = []
chunk_id = 0

for json_file in jsons:
    with open(f"jsons/{json_file}") as f:
        content = json.load(f)
    print(f"Creating Embeddings for {json_file}")
    embeddings = create_embedding([c['text'] for c in content['chunks']])

    for i, chunk in enumerate(content['chunks']):
        chunk['chunk_id'] = chunk_id
        chunk['embedding'] = embeddings[i]
        chunk_id += 1
        my_dicts.append(chunk)
    # print(my_dicts)

# df = pd.DataFrame.from_records(my_dicts)
# joblib.dump(df,'embeddings.joblib')
# a = create_embedding(["Cat sat on the mat", "Harry dances on a mat"])
# print(a)

df = pd.DataFrame.from_records(my_dicts)

client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="course_chunks",
    metadata={"hnsw:space": "cosine"},
)

batch_size = 500
for i in range(0, len(df), batch_size):
    batch = df.iloc[i:i + batch_size]
    collection.upsert(
        ids=[str(x) for x in batch["chunk_id"]],
        embeddings=batch["embedding"].tolist(),
        documents=batch["text"].tolist(),
        metadatas=batch[["title", "number", "start", "end"]].to_dict(orient="records"),
    )

print("Stored chunks:", collection.count())

