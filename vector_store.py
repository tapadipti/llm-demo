import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
import json

with open("docs.json", "r") as f:
    docs = json.load(f)

with open("metadatas.json", "r") as f:
    metadatas = json.load(f)

print(f"Processing {len(docs)} documents.")

emb = OpenAIEmbeddings()
emb.show_progress_bar = True

# Here we create a vector store from the documents and save it to disk.
store = FAISS.from_texts(docs, emb, metadatas=metadatas)
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)
