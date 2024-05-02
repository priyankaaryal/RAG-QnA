from io import StringIO

from streamlit.runtime.uploaded_file_manager import UploadedFile


def extract_full_text(uploaded_file: UploadedFile) -> str:
    """
    Extracts and returns the text content from a .txt uploaded file. Raises an error for unsupported file formats.

    Args:
        uploaded_file (UploadedFile): The file uploaded by the user, expected to be a text file.

    Returns:
        str: The text extracted from the file, if it's a .txt format.

    Raises:
        NotImplementedError: If the file is a PDF, as PDF reading is not supported.
        TypeError: If the file format is neither .txt nor .pdf, indicating an unsupported file type.
    """
    if uploaded_file.name.endswith("txt"):
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        full_text = stringio.read()
    elif uploaded_file.name.endswith("pdf"):
        raise NotImplementedError("Can't read PDF files yet")
    else:
        raise TypeError(f"Cannot parse file {uploaded_file.name}")
    return full_text
