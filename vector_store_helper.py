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
            surrounding_chunks: int = 0
    ) -> Tuple[List[str], List[str], List[str]]:
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_n
        )
        retrieved_ids = [int(i) for i in results["ids"][0]]
        context_ids_to_retrieve = []
        for id_int in retrieved_ids:
            for context_id in range(id_int - surrounding_chunks, id_int + surrounding_chunks + 1):
                if 0 <= context_id < self.n_docs:
                    context_ids_to_retrieve.append(str(context_id))
        retrieved_results = self.collection.get(ids=context_ids_to_retrieve)
        return retrieved_results["ids"], retrieved_results["documents"], retrieved_results["metadatas"]
