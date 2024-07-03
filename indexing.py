import os
from typing import Optional

from llama_index.core import (
    DocumentSummaryIndex,
    SummaryIndex,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.storage import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding


class IndexBuilder:
    """
    索引構建器
    """

    @staticmethod
    async def build_index(
        index_class: BaseIndex,
        persist_path: str,
        model: str,
        nodes: Optional[list] = None,
        rebuild: bool = False,
    ):
        """
        構建索引並且儲存，如果索引已經存在，則直接加載索引。
        """
        if not os.path.exists(persist_path) or rebuild:
            index = index_class(nodes, embed_model=OpenAIEmbedding(model=model))
            index.storage_context.persist(persist_dir=persist_path)
        else:
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=persist_path)
            )
        return index

    @staticmethod
    async def build_json_index(
        persist_path: str, model: str = "text-embedding-3-large", doc: Document = None
    ) -> VectorStoreIndex:
        nodes = doc
        # if doc:
        #     nodes = JSONNodeParser().get_nodes_from_documents(doc)
        return await IndexBuilder.build_index(
            VectorStoreIndex, persist_path, model, nodes, rebuild=True
        )

    @staticmethod
    async def build_vector_index(
        persist_path: str,
        model: str = "text-embedding-3-large",
        doc: Document = None,
    ) -> VectorStoreIndex:
        nodes = None
        if doc:
            splitter = SentenceSplitter(chunk_size=2048, chunk_overlap=512)
            nodes = splitter.get_nodes_from_documents(doc)
        return await IndexBuilder.build_index(
            VectorStoreIndex, persist_path, model, nodes
        )

    @staticmethod
    async def build_doc_summary_index(
        persist_path: str, model: str = "text-embedding-3-large", doc: Document = None
    ) -> DocumentSummaryIndex:
        nodes = None
        if doc:
            splitter = SentenceSplitter(chunk_size=1024)
            nodes = splitter.get_nodes_from_documents(doc)
        return await IndexBuilder.build_index(
            DocumentSummaryIndex, persist_path, model, nodes
        )

    @staticmethod
    async def build_all_doc_summary_index(
        persist_path: str, model: str = "text-embedding-3-large", doc: Document = None
    ) -> SummaryIndex:
        nodes = None
        if doc:
            splitter = SentenceSplitter(chunk_size=1024)
            nodes = splitter.get_nodes_from_documents(doc)
        return await IndexBuilder.build_index(SummaryIndex, persist_path, model, nodes)
