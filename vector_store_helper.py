from typing import List, Tuple

import chromadb


class VectorStoreHelper:
    def __init__(self):
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.create_collection(name="rag_store",
                                                        get_or_create=True)
        self.n_docs = 0

    def add_embeddings_to_collection(
            self,
            embeddings: List[List[float]],
            documents: List[str],
            metadatas: List[dict] = None
    ):
        """
        Adds a batch of embeddings, documents, and optional metadata to the collection and updates the document count.

        Args:
            embeddings (List[List[float]]): The embeddings for the documents to be stored.
            documents (List[str]): The documents associated with the embeddings.
            metadatas (List[dict], optional): Additional metadata for each document.

        Returns:
            None
        """
        self.collection.add(
            ids=[str(self.n_docs + i) for i in range(len(documents))],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )
        self.n_docs += len(documents)

    def query_collection(
            self,
            query_embedding: List[float],
            top_n: int = 3,
            surrounding_docs: int = 0
    ) -> List[Tuple[str, str, dict[str, str]]]:
        """
        Queries the collection with a given embedding and retrieves the top N similar documents and their surrounding context.

        Args:
            query_embedding (List[float]): The embedding vector to query against the collection.
            top_n (int, optional): Number of top documents to retrieve. Defaults to 3.
            surrounding_docs (int, optional): Number of documents to retrieve around each top document for context.

        Returns:
            List[Tuple[str, str, dict[str, str]]]: A list of tuples, each containing the document ID, document text, and metadata.
        """
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_n)
        retrieved_ids = [int(i) for i in results["ids"][0]]
        context_ids_to_retrieve = set()
        for id_int in retrieved_ids:
            for context_id in range(id_int - surrounding_docs, id_int + surrounding_docs + 1):
                if 0 <= context_id < self.n_docs:
                    context_ids_to_retrieve.add(str(context_id))
        context_ids_to_retrieve = list(context_ids_to_retrieve)
        retrieved_results = self.collection.get(ids=context_ids_to_retrieve)
        retrieved_results = zip(retrieved_results["ids"], retrieved_results["documents"],
                                retrieved_results["metadatas"])
        return sorted(retrieved_results, key=lambda x: int(x[0]))
