import streamlit as st
import doc_search

def main():
    loaded_files = []

    st.title("Document Search App")

    # File uploader
    uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)

    # Load the data from files
    if st.button("Load Data"):
        if uploaded_files is not None:
            for file in uploaded_files:
                 retriever = doc_search.load_data(file)

    # Search bar
    search_query = st.text_input("Enter your search query")

    # Search button
    if st.button("Search"):
        print(doc_search.answer_query(retriever, search_query))

if __name__ == "__main__":
    main()
