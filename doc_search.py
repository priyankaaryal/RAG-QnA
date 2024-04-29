from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

def load_data(filename):
    loader = TextLoader(filename)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(data)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(docs, embeddings)
    #print(f"{filename.name} loaded")
    print(filename, "loaded")
    return db

def answer_query(retriever, query):
    relevant_answers = []

    for answer in retriever.similarity_search_with_score(query)[:3]:
        rel_answer = answer[0]
        score = answer[1]
        relevant_answers.append((rel_answer, score))

        print("Answer: ", answer)
        print("Similarity Score: ", score)
        print("\n")

    return relevant_answers

