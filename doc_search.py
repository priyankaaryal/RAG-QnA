import streamlit as st
from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

OPENAI_MODEL = "gpt-3.5-turbo"
SINGLE_QUERY_PROMPT = "Based on the given query and the relevant answers, choose the best suiting answer for the query for the user and reword it so the answer is coherent. DO NOT add any other information to the answer."


def invoke_llm(prompt):
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def load_data(data: list[str]) -> Chroma:
    # todo: have some rationale behind chunk size
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(data)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings, ids=[str(i) for i in range(len(docs))])
    print("Data loaded successfully")

    return db


def answer_single_query(retriever, query, threshold):
    relevant_answers = []

    for answer in retriever.similarity_search_with_score(query)[:3]:
        rel_answer = answer[0]
        score = answer[1]
        if score <= threshold:
            relevant_answers.append((rel_answer, score))

            print("Answer: ", answer)
            print("Similarity Score: ", score)
            print("\n")

    prompt = SINGLE_QUERY_PROMPT + f"\n Query: {query} \n Answers: {relevant_answers}"
    final_answer = invoke_llm(prompt)

    return final_answer


def answer_chat_query(retriever, messages):
    context = retriever.similarity_search_with_score(messages[-1]["content"])[:3]
    print(context)
    messages[0]["content"] = f"""You are RAG-GPT. 
        All your answers should be based on the context between the <context></context> tags below. 
        DO NOT add any other information to the answer. Never offer to search beyond the confines of your knowledge.
        
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
