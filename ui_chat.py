import streamlit as st
from langchain_text_splitters import CharacterTextSplitter

import doc_search
import file_parsing
from doc_search import embedder
from vector_store_helper import VectorStoreHelper

def store_in_vector_store(document: str, document_name: str):
    """
    Processes and stores a document in the vector store by splitting it into chunks, embedding these chunks,
    and then adding them to the collection with metadata.

    Args:
        document (str): The document text to be processed and stored.
        document_name (str): The name of the document, used in the metadata for identification.

    Returns:
        None: This function does not return anything but prints a success message upon completion.
    """
    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    chunks = [doc.page_content for doc in text_splitter.create_documents([document])]
    metadatas = [{"document_name": document_name, "chunk_id": i + 1} for i in range(len(chunks))]
    embeddings = embedder.embed_documents(texts=chunks)
    st.session_state["retriever"].add_embeddings_to_collection(documents=chunks,
                                                               embeddings=embeddings,
                                                               metadatas=metadatas)
    print("Data loaded successfully")

st.title("Document Search App")

# Initialize variables
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = VectorStoreHelper()
if 'files_processed' not in st.session_state:
    st.session_state['files_processed'] = set()
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": ""}]

# File uploader
uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state['files_processed']:
            st.session_state['files_processed'].add(uploaded_file.name)
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
