from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def invoke_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def load_data(data):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.create_documents(data)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    print("Data loaded successfully")

    return db

def answer_query(retriever, query, threshold):
    relevant_answers = []
    prompt = "Based on the given query and the relevant answers, choose the best suiting answer for the query for the user and reword it so the answer is coherent. DO NOT add any other information to the answer."

    for answer in retriever.similarity_search_with_score(query)[:3]:
        rel_answer = answer[0]
        score = answer[1]
        if score <= threshold:
            relevant_answers.append((rel_answer, score))

            print("Answer: ", answer)
            print("Similarity Score: ", score)
            print("\n")

    prompt += f"\n Query: {query} \n Answers: {relevant_answers}"
    final_answer = invoke_llm(prompt)

    return final_answer
