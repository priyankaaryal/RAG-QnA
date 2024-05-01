from io import StringIO

from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_full_text(uploaded_file: UploadedFile) -> str:
    if uploaded_file.name.endswith("txt"):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        full_text = stringio.read()
    elif uploaded_file.name.endswith("pdf"):
        raise NotImplementedError("Can't read pdfs yet")
    else:
        raise TypeError(f"Cannot parse file {uploaded_file.name}")
    return full_text
