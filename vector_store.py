import faiss
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
import pickle
import json
import dvc.api


params = dvc.api.params_show()['OpenAIEmbeddings']
print(params)

with open("docs.json", "r") as f:
    docs = json.load(f)

with open("metadatas.json", "r") as f:
    metadatas = json.load(f)

docs = docs[0:1000]
metadatas = metadatas[0:1000]

print(f"Processing {len(docs)} documents.")

try:
    emb = OllamaEmbeddings()
except:
    print('Could not create embeddings object')
print('Going to show progress now')
emb.show_progress = True

print('Going to try from texts')
# Here we create a vector store from the documents and save it to disk.
try:
    store = FAISS.from_texts(docs, emb, metadatas=metadatas)
except:
    print('Could not create embeddings from texts')
print('Going to write now')
faiss.write_index(store.index, "docs.index")
store.index = None
with open("faiss_store.pkl", "wb") as f:
    pickle.dump(store, f)

