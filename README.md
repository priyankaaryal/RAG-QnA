# Document Search App

## Description
The Document Search App allows users to upload their documents and query specific information within those documents. It provides a chat-based web interface, powered by RAG, enabling users to upload a file, parse it, and ask questions relevant to the documents. 

With the help of ChromaDB, the application stores embeddings and relevant metadata and performs vector searches to retrieve relevant documents effectively. It utilizes GPT to refine answers, recontextualize user queries to align with the entire conversation, and provide accurate responses to queries.

## Installation
To get started with this project, you need to install the required Python packages. Run the following command in your terminal:
```bash
pip install -r requirements.txt
```
*Note: Python version 3.10 was used to develop this app.*

## Usage
To run the project after installing the dependencies, navigate to the project directory and run the streamlit command:
```bash
streamlit run ui_chat.py
```
After running the command, the streamlit app will automatically start in your browser, or you can view by clicking the Local URL provided in the terminal. 

## Development Process
This section details the step-by-step approach and methodologies used in the development of the Document Search App. The process reflects a combination of iterative development, integration of different technologies, and continuous refinement based on testing.

### Initial Setup and Planning
**Skeletal UI Creation:** Utilized Streamlit to create a basic user interface that allows for document uploads and query inputs. This step established the framework for subsequent features and integrations.

### Core Functionality Development

**Vector Search Implementation:** Developed the core functionality of vector search, storing embedded chunks of uploaded documents in ChromaDB, and executing a similarity search to identify the top n relevant documents.

**Language Model:** Employed a language model (GPT) to improve the coherence of answers and to select the best response from the top n documents extracted from the similarity search.

**UI and Logic Integration:** Integrated the UI web app with the backend vector search and language model logic to ensure seamless operation between the frontend and backend components.

### Enhancements and Optimizations

**UI enhancement:** Incorporated a chat interface to the UI that allows users to ask follow-up questions regarding the uploaded documents.

**Recontextualization:** Recontextualized the user prompt based on message history before performing vector search.

**Surrounding context:** Added logic to retrieve surrounding context during vector search, ensuring that the information retrieved is complete and not truncated arbitrarily.

**Source Attribution / Grounding:**  Included metadata such as file name and page number in search results to prevent hallucination and provide attribution to source.

### Documentation and Testing

**Documentation Addition:** Ensured comprehensive documentation for each function and added a README file.

**Test Data:** Two different domains of test data were used to test the application.

## Problems Encountered and Solutions

**Out-Of-Context Questions:** Performing vector search using only the latest user prompt would not retrieve relevant documents, preventing GPT from responding accurately. The image below illustrates this:

![Alt text](app_images/fish_1.png?raw=true "no-context-question")

To address this issue, the user's latest message was rephrased using the chat history to reflect the context of previous messages. This rephrased message was then used to query the vector database to ensure the semantic retrieval of the most relevant documents. The prompt below was used to re-contextualize the user query:
```commandline
RECONTEXTUALIZATION_PROMPT = """Please return the last user message in the following chat history (Actual Chat History) so that it is a complete sentence reflecting the context of the previous messages.
The sentence should make sense on its own and any pronouns should be replaced by the object they're referring to. 
Respond with only the recontextualized sentence. Pay more attention to more recent messages.

Example

Chat history:
user: What is beef stew?
assistant: Beef stew is a hearty, traditional dish made by slow-cooking chunks of beef along with vegetables and a flavorful liquid, typically water or stock.
user: How to make it?

Recontextualized last message:
How to make beef stew?

Actual Chat History:

"""
```

After rephrasing, the user message becomes "How to make fish tacos?". We then use this new string (successfully) to query the vector database. 
Result after re-contextualization of the user message:

![Alt text](app_images/fish_2.png?raw=true "no-context-question")

**Surrounding Context Retrieval:** A smaller chunk size might result in better correlation with the user query. However, smaller chunk size also means the retrieved information from vector search may be incomplete. To ensure that no information is lost during the retrieval process, the retrieval of surrounding context was implemented. 

Langchain-Chroma does not support retrieving consecutive chunks of a document. So, the official ChromaDB library was used instead.

![Alt text](app_images/vectorstore.png?raw=true "no-context-question")

Each chunk loaded in the ChromaDB vectorstore was assigned a document_id which was stored in order. 

During the retrieval process, n-documents were retrieved along with the specified number of surrounding chunks ensuring no information was lost. 

## App Demo 
You can watch a short demo of this app using the link below:

[Demo Video](https://drive.google.com/file/d/1nlUHhHOy_B6bR2kmmWxrhUpErQhsVGck/view?usp=sharing/view)
