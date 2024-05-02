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
            surrounding_docs: int = 0
    ) -> List[Tuple[str, str, dict[str, str]]]:
        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_n
        )
        retrieved_ids = [int(i) for i in results["ids"][0]]
        context_ids_to_retrieve = set()
        for id_int in retrieved_ids:
            for context_id in range(id_int - surrounding_docs, id_int + surrounding_docs + 1):
                if 0 <= context_id < self.n_docs:
                    context_ids_to_retrieve.add(str(context_id))
        context_ids_to_retrieve = list(context_ids_to_retrieve)
        retrieved_results = self.collection.get(ids=context_ids_to_retrieve)
        retrieved_results = zip(retrieved_results["ids"], retrieved_results["documents"], retrieved_results["metadatas"])
        return sorted(retrieved_results, key=lambda x: int(x[0]))
