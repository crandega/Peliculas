import pandas as pd
from sentence_transformers import SentenceTransformer, util

df = pd.read_csv(r'D:\Documents\semantic_search\IMDB top 1000.csv')
df.head()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

embeddings = model.encode(df['Description'].tolist(), batch_size=64, show_progress_bar=True)

df['embeddings'] = embeddings.tolist()

def compute_similarity(example, query_embedding):
    embedding = example['embeddings'] 
    similarity = util.cos_sim(embedding, query_embedding).item()
    return similarity  

query_embedding = model.encode(['a travel time adventure'])[0]
df['similarity'] = df.apply(lambda x: compute_similarity(x, query_embedding), axis=1)
df = df.sort_values(by='similarity', ascending=False)

df.head()['Title']