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
# OPENAI_MODEL = "gpt-4-turbo"


def invoke_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def answer_chat_query(retriever: VectorStoreHelper, messages: list[dict[str, str]]) -> list[Any] | str:
    query_embedding = embedder.embed_query(messages[-1]["content"])
    retrieved_results = retriever.query_collection(query_embedding, top_n=5, surrounding_docs=1)
    print(retrieved_results)
    context = [f"----\nContext Page Start\nfile_name: {metadata['document_name']}\npage_number: {metadata['chunk_id']}\n{doc}\nContext Page End\n----\n" for _, doc, metadata in retrieved_results]
    messages[0]["content"] = f"""You are RAG-GPT. You must assist the user with responses that answer their query, based on given context.
        All your answers should be based on the context between the <context></context> tags below. 
        DO NOT add any other information to the answer. Never offer to search beyond the confines of the context given below. Simply say that the answer does not exist in your context.
        Ensure proper markdown formatting with adequate whitespace and linebreaks.
        If and only if the answer is in the context below, you must output your answer, the document name and chunk ID used for your answer at the end in the following format.
        
        <Your Answer>
        
        ----
        
        File Name: <file_name of the relevant context page>
        
        Page Number: <page_number of the relevant context page>
        
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
