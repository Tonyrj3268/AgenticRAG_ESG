from typing import Optional

from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import BaseNode, NodeWithScore
from pydantic import Field


class PageGroupPostprocessor(BaseNodePostprocessor):
    all_nodes_from_doc: list[BaseNode] = Field(default_factory=list)

    def _postprocess_nodes(
        self, nodes: list[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> list[NodeWithScore]:
        # 創建頁碼到分數的映射
        page_score_map = {node.node.metadata.get("pages"): node.score for node in nodes}

        # 從all_nodes中篩選出相關頁碼的節點並創建NodeWithScore對象
        result_nodes = [
            NodeWithScore(
                node=node, score=page_score_map.get(node.metadata.get("pages"), 0)
            )
            for node in self.all_nodes_from_doc
            if node.metadata.get("pages") in page_score_map
        ]

        # 按頁碼排序結果
        return sorted(result_nodes, key=lambda x: x.node.metadata.get("pages"))
