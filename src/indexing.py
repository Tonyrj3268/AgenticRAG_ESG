import os
from typing import Optional, Union

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.indices.base import BaseIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document
from llama_index.core.storage import StorageContext
from llama_index.embeddings.openai import OpenAIEmbedding


class IndexBuilder:
    """
    索引構建器
    """

    async def build_index(
        self,
        index_class: BaseIndex,
        persist_path: str,
        data: Optional[Union[list[Document], list[BaseNode]]] = None,
    ):
        """
        構建索引並且儲存，如果索引已經存在，則直接加載索引。
        """
        if not os.path.exists(persist_path) or data is not None:
            index = index_class(data, use_async=True)
            index.storage_context.persist(persist_dir=persist_path)
        else:
            index = load_index_from_storage(
                StorageContext.from_defaults(persist_dir=persist_path)
            )
        return index

    async def build_json_index(
        self, persist_path: str, data: Optional[Document] = None
    ) -> VectorStoreIndex:
        # TODO: Implement JSONNodeParser if needed
        # if doc:
        #     nodes = JSONNodeParser().get_nodes_from_documents([doc])
        return await self.build_index(VectorStoreIndex, persist_path, data)

    async def build_vector_index(
        self,
        persist_path: str,
        data: Optional[list[Document]] = None,
    ) -> VectorStoreIndex:
        if data:
            splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=256)
            data = splitter.get_nodes_from_documents(data)

        return await self.build_index(VectorStoreIndex, persist_path, data)
