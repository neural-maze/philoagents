from langchain.tools.retriever import create_retriever_tool
from langchain_core.documents import Document

from philoagents.application.rag.retrievers import get_retriever
from philoagents.config import settings
from philoagents.infrastructure.mongo.client import MongoClientWrapper
from philoagents.infrastructure.mongo.indexes import MongoIndex

retriever = get_retriever(
    embedding_model_id=settings.RAG_TEXT_EMBEDDING_MODEL_ID,
    k=settings.RAG_TOP_K,
    device=settings.RAG_DEVICE,
)
# retriever.pre_filter = {"philosopher_id": {"$eq": "plato"}}


retriever_tool = create_retriever_tool(
    retriever,
    "retrieve_philosopher_context",
    "Search and return information about a specific philosopher. Always use this tool when the user asks you about a philosopher, their works, ideas or historical context.",
)

tools = [retriever_tool]

if __name__ == "__main__":
    # with MongoClientWrapper(
    #     model=Document, collection_name=settings.MONGO_LONG_TERM_MEMORY_COLLECTION
    # ) as client:
    #     index = MongoIndex(
    #         retriever=retriever,
    #         mongodb_client=client,
    #     )
    #     index.create(
    #         is_hybrid=True, embedding_dim=settings.RAG_TEXT_EMBEDDING_MODEL_DIM
    #     )

    results = retriever_tool.invoke({"query": "What is the meaning of life?"})
    print(results)
