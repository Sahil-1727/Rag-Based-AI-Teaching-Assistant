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
def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={
        # "model": "deepseek-r1",
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    })

    response = r.json()
    print(response)
    return response

df = joblib.load('embeddings.joblib')
incoming_query = input("Ask a Question: ")
question_embedding = create_embedding([incoming_query])[0]

# Find similarities of question_embedding with other embeddings
# print(np.vstack(df['embedding'].values))
# print(np.vstack(df['embedding']).shape)
similarities = cosine_similarity(np.vstack(df['embedding']), [question_embedding]).flatten()
# print(similarities)
top_results = 5
max_indx = similarities.argsort()[::-1][0:top_results]
# print(max_indx)
new_df = df.loc[max_indx]
print(new_df[["title", "number", "text"]])

prompt = f'''You are an AI Teaching Assistant for the "Sigma Web Development" course.

Here are the relevant video transcript chunks based on the user's query. Each chunk contains the video title, video number, start time in seconds, end time in seconds, and the spoken text:
{new_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
----------------------------------
User Query: {incoming_query}
----------------------------------
Instructions:
1. Answer the user's question based strictly on the provided video chunks.
2. Explicitly state where the content is taught (mention the video number, title, and exact timestamps).
3. Guide the user to go to that particular video.
4. If the user asks an unrelated question that cannot be answered using the provided chunks, politely inform them that you can only answer questions related to the course videos.
'''

# for index, item in new_df.iterrows():
#     print(index, item["title"], item["number"], item["text"], item["start"], item["end"])

with open("prompt.txt", "w") as f:
    f.write(prompt)

response = inference(prompt)["response"]
print(response)

with open("response.txt", "w") as f:
    f.write(response)
