import streamlit as st

def load_data(file):
    print(f"{file.name} loaded")
    return NotImplementedError

def main():
    loaded_files = []

    st.title("Document Search App")

    # File uploader
    uploaded_files = st.file_uploader("Upload file/files", accept_multiple_files=True)

    # Load the data from files
    if st.button("Load Data"):
        if uploaded_files is not None:
            for file in uploaded_files:
                 load_data(file)

    # Search bar
    search_query = st.text_input("Enter your search query")

    # Search button
    if st.button("Search"):
        return NotImplementedError

if __name__ == "__main__":
    main()
