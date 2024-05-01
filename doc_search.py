from typing import Any

import streamlit as st
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
import os

from vector_store_helper import VectorStoreHelper

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
embedder = OpenAIEmbeddings()

OPENAI_MODEL = "gpt-3.5-turbo"


def invoke_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def answer_chat_query(retriever: VectorStoreHelper, messages: list[dict[str, str]]) -> list[Any] | str:
    doc_ids, docs, metadatas = retriever.query_collection(embedder.embed_query(messages[-1]["content"]))
    print(doc_ids)
    print(docs)
    print(metadatas)
    context = [f"----\nContext Chunk: {doc}\nfile_name: {metadata['document_name']}\nchunk_id: {metadata['chunk_id']}\n----\n" for doc, metadata in zip(docs, metadatas)]
    messages[0]["content"] = f"""You are RAG-GPT. 
        All your answers should be based on the context between the <context></context> tags below. 
        DO NOT add any other information to the answer. Never offer to search beyond the confines of your knowledge.
        If and only if the answer is in the context below, you must output the document name and chunk ID used for your answer at the end in the following format:
        
        ----
        
        File Name: file_name of the relevant context chunk
        
        Chunk ID: chunk_id
        
        ----
        <context>
        {context}
        </context>"""
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model=OPENAI_MODEL,
            stream=True,
        )
        return st.write_stream(stream)
