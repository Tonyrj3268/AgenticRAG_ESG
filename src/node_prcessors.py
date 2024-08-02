from typing import Optional

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import BaseNode
from pydantic import Field


class PageGroupPostprocessor(BaseNodePostprocessor):
    all_nodes_from_doc: list[BaseNode] = Field(default_factory=list)

    def _postprocess_nodes(
        self, nodes: list[BaseNode], query_bundle: Optional[QueryBundle]
    ) -> list[BaseNode]:
        # 使用set收集所有相關的頁碼
        relevant_pages = set(node.metadata.get("pages") for node in nodes)

        # 從all_nodes中篩選出相關頁碼的節點
        result_nodes = [
            node
            for node in self.all_nodes_from_doc
            if node.metadata.get("pages") in relevant_pages
        ]

        # 按頁碼排序結果
        return sorted(result_nodes, key=lambda x: x.metadata.get("pages"))
