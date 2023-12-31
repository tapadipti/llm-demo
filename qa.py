"""Ask a question to the notion database."""
import faiss
from langchain.llms import Ollama
from langchain.chains import RetrievalQAWithSourcesChain
import os
import pickle
import pandas as pd
import dvc.api


params = dvc.api.params_show()
chat_params = params['ChatOpenAI']
qa_params = params['Retrieval']
print(chat_params)
print(qa_params)

# Load the LangChain.
index = faiss.read_index("docs.index")

with open("faiss_store.pkl", "rb") as f:
    store = pickle.load(f)

df = pd.read_csv("canfy.csv")
sample_questions = df["Q"].to_list()[0:2]

store.index = index
llm = Ollama(model="llama2")
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm,
                                                    retriever=store.as_retriever(), max_tokens_limit=qa_params['max_tokens_limit'],
                                                    reduce_k_below_max_tokens=qa_params['reduce_k_below_max_tokens'],
                                                    verbose=qa_params['verbose'])

records = []
for question in sample_questions:
    question = question.strip()
    print(f"Question: {question}")

    result = chain({"question": question})
    records.append({"Q": question, "A": result["answer"].strip(), "sources": result['sources'].strip()})

    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")
    print("")

df = pd.DataFrame.from_records(records)
df.to_csv("results.csv", header=True, index=False)
