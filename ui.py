import streamlit as st
import doc_search

def main():
    st.title("Document Search App")

    if 'loaded_files' not in st.session_state:
        st.session_state['loaded_files'] = []
    if 'retriever' not in st.session_state:
        st.session_state['retriever'] = None

    # File uploader
    uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)

    # Load the data from files
    if st.button("Load Data"):
        if uploaded_files is not None:
            for file in uploaded_files:
                file_content = file.read().decode()
                st.session_state['loaded_files'].append(file_content)
                st.write(f"Loaded {len(file_content)} characters from {file.name}.")

            st.session_state['retriever'] = doc_search.load_data(st.session_state['loaded_files'])
            st.write(f"{file.name} loaded successfully")

    # Search bar
    search_query = st.text_input("Enter your search query")

    # Search button
    if st.button("Search"):
        if st.session_state['retriever'] and search_query:
            results = doc_search.answer_query(st.session_state['retriever'], search_query)
            st.write(results)
        elif st.session_state['retriever'] and not search_query:
            st.write("Please enter the query")
        elif not st.session_state['retriever'] and search_query:
            st.write("Please load the file first")
        else:
            st.write("Please load the file and enter the query")

if __name__ == "__main__":
    main()
