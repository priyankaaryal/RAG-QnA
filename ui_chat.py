import streamlit as st
from langchain_text_splitters import CharacterTextSplitter

import doc_search
import file_parsing
from doc_search import embedder
from vector_store_helper import VectorStoreHelper

st.title("Document Search App")

# Initialize variables
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = VectorStoreHelper()
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": ""}]

# File uploader
uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)


# Load the data from files
def store_in_vector_store(document: str, document_name: str):
    # todo: have some rationale behind chunk size
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = [doc.page_content for doc in text_splitter.create_documents([document])]
    metadatas = [{"document_name": document_name, "chunk_id": i+1} for i in range(len(chunks))]
    embeddings = embedder.embed_documents(texts=chunks)
    st.session_state["retriever"].add_embeddings_to_collection(documents=chunks,
                                                               embeddings=embeddings,
                                                               metadatas=metadatas)
    print("Data loaded successfully")


if uploaded_files:
    for uploaded_file in uploaded_files:
        file_content = file_parsing.extract_full_text(uploaded_file)
        st.write(f"{uploaded_file.name} loaded successfully")
        store_in_vector_store(document=file_content, document_name=uploaded_file.name)

    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your messages here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = doc_search.answer_chat_query(st.session_state.retriever, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
