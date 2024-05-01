import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os

from vector_store_helper import VectorStoreHelper

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
embedder = OpenAIEmbeddings()

OPENAI_MODEL = "gpt-3.5-turbo"
SINGLE_QUERY_PROMPT = "Based on the given query and the relevant answers, choose the best suiting answer for the query for the user and reword it so the answer is coherent. DO NOT add any other information to the answer."


def invoke_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def load_data(data: list[str]) -> VectorStoreHelper:
    # todo: have some rationale behind chunk size
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = [doc.page_content for doc in text_splitter.create_documents(data)]
    db = VectorStoreHelper()
    embeddings = embedder.embed_documents(texts=docs)
    db.add_embeddings_to_collection(documents=docs, embeddings=embeddings)
    print("Data loaded successfully")

    return db


def answer_single_query(retriever: VectorStoreHelper, query: str, threshold: float) -> str:
    doc_ids, docs = retriever.query_collection(embedder.embed_query(query))
    print(doc_ids)
    print(docs)

    prompt = SINGLE_QUERY_PROMPT + f"\n Query: {query} \n Answers: {docs}"
    final_answer = invoke_llm(prompt)

    return final_answer


def answer_chat_query(retriever, messages):
    doc_ids, docs = retriever.query_collection(embedder.embed_query(messages[-1]["content"]))
    print(doc_ids)
    print(docs)
    messages[0]["content"] = f"""You are RAG-GPT. 
        All your answers should be based on the context between the <context></context> tags below. 
        DO NOT add any other information to the answer. Never offer to search beyond the confines of your knowledge.
        
        <context>
        {docs}
        </context>"""
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model=OPENAI_MODEL,
            stream=True,
        )
        return st.write_stream(stream)
