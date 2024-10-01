import os
from typing import Any

import streamlit as st
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI

from vector_store_helper import VectorStoreHelper

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)
embedder = OpenAIEmbeddings()

# OPENAI_MODEL = "gpt-3.5-turbo"
OPENAI_MODEL = "gpt-4o"

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


def invoke_llm(prompt: str) -> str:
    """
    Invoke the language model to generate a completion based on the given prompt.

    Args:
        prompt (str): The prompt for the language model.

    Returns:
        str: The generated completion as a string.
    """
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()


def create_recontextualization_prompt(messages: list[dict[str, str]]) -> str:
    """
    Create a recontextualization prompt based on the given messages.

    Args:
        messages (list[dict[str, str]]): List of messages in the chat history.

    Returns:
        str: The populated recontextualization prompt as a string.
    """
    prompt_list = [RECONTEXTUALIZATION_PROMPT]
    for message in messages[1:]:
        prompt_list.append(f"{message['role']}: {message['content']}")
    return "\n".join(prompt_list)


def answer_chat_query(retriever: VectorStoreHelper, messages: list[dict[str, str]]) -> list[Any] | str:
    """
    Answer a chat query based on the given messages and RAG-retrieved context.

    Args:
        retriever (VectorStoreHelper): The retriever object.
        messages (list[dict[str, str]]): List of messages in the chat history.

    Returns:
        list[Any] | str: The response to the chat query.
    """

    # Recontextualize the latest user message, and find the most similar document chunks
    recontextualized_last_message = invoke_llm(create_recontextualization_prompt(messages))
    query_embedding = embedder.embed_query(recontextualized_last_message)
    retrieved_results = retriever.query_collection(query_embedding, top_n=5, surrounding_docs=1)

    context = [
        f"----\nContext Page Start\nfile_name: {metadata['document_name']}\npage_number: {metadata['chunk_id']}\n{doc}\nContext Page End\n----\n"
        for _, doc, metadata in retrieved_results]

    # Put context into system prompt
    messages[0]["content"] = f"""You are RAG-GPT. You must assist the user with responses that answer their query, based on given context.
        All your answers should be based on the context between the <context></context> tags below. 
        DO NOT add any other information to the answer. Never offer to search beyond the confines of the context given below. Simply say that the answer does not exist in your context.
        Ensure proper markdown formatting with adequate whitespace and linebreaks.
        If and only if the answer is in the context below, you must output your answer, the document name and chunk ID used for your answer at the end in the following format.
        
        <Your Answer>
        
        ----
        
        File Name: <file_name of the relevant context page>
        
        Page Number: <page_number of the relevant context page>
        
        Excerpt: <relevant context where the answer is generated from>
        
        ----
        
        <context>
        {context}
        </context>"""
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            messages=messages,
            model=OPENAI_MODEL,
            stream=True,
        )
        return st.write_stream(stream)
