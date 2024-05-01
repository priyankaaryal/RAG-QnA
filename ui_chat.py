import streamlit as st

import doc_search

st.title("Document Search App")

# Initialize variables
if 'loaded_files' not in st.session_state:
    st.session_state['loaded_files'] = []
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": ""}]

# File uploader
uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)

# Load the data from files
if st.button("Load Data"):
    if uploaded_files:
        for file in uploaded_files:
            # todo: read one file at a time
            file_content = file.read().decode()
            st.session_state['loaded_files'].append(file_content)
            st.write(f"{file.name} loaded successfully")

        st.session_state['retriever'] = doc_search.load_data(st.session_state['loaded_files'])
    else:
        st.write("Please select a file to load first")

if retriever := st.session_state['retriever']:

    for message in st.session_state.messages[1:]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Your messages here"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = doc_search.answer_chat_query(st.session_state.retriever, st.session_state.messages)
        st.session_state.messages.append({"role": "assistant", "content": response})
