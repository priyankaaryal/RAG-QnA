from openai import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_data(data):
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.create_documents(data)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    print("Data loaded successfully")

    return db

def answer_query(retriever, query, threshold):
    relevant_answers = []

    for answer in retriever.similarity_search_with_score(query)[:3]:
        rel_answer = answer[0]
        score = answer[1]
        if score <= threshold:
            relevant_answers.append((rel_answer, score))

        print("Answer: ", answer)
        print("Similarity Score: ", score)
        print("\n")

    return relevant_answers

