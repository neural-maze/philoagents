from langchain_mongodb.index import create_fulltext_search_index

from .client import MongoClientWrapper


class MongoIndex:
    def __init__(
        self,
        retriever,
        mongodb_client: MongoClientWrapper,
    ) -> None:
        self.retriever = retriever
        self.mongodb_client = mongodb_client

    def create(
        self,
        embedding_dim: int,
        is_hybrid: bool = False,
    ) -> None:
        vectorstore = self.retriever.vectorstore

        vectorstore.create_vector_search_index(
            dimensions=embedding_dim,
            filters=[{"type": "filter", "path": "philosopher_id"}],
            update=True,
        )

        if is_hybrid:
            create_fulltext_search_index(
                collection=self.mongodb_client.collection,
                field=vectorstore._text_key,
                index_name=self.retriever.search_index_name,
            )
