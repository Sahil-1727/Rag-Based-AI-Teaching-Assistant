import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
import joblib

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

df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 30
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
print(new_df[["title", "number", "text"]])

for index, item in new_df.iterrows():
    print(index, item["title"], item["number"], item["text"], item["start"], item["end"])