import asyncio
import os

from llama_index.agent.openai import OpenAIAgent, OpenAIAgentWorker
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from indexing import IndexBuilder
from prompts import (
    INDUSTRY_AGENT_PROMPT,
    INDUSTRY_AGENT_PROMPT_EN,
    NOTES_AGENT_PROMPT,
    NOTES_AGENT_PROMPT_EN,
)


class AgentBuilder:
    def __init__(self, base_dir: str, esg_dir_path: str):
        self.base_dir = base_dir
        self.esg_dir_path = esg_dir_path

    # Build industry and note agent
    async def build_industry_note_agent(self, doc) -> tuple[OpenAIAgent, OpenAIAgent]:
        vector_index = await IndexBuilder.build_json_index(
            f"{self.esg_dir_path}/industry_map", doc=doc, model="text-embedding-3-small"
        )
        vector_query_engine = vector_index.as_query_engine(response_mode="compact")

        # 建立 agent
        industry_agent = OpenAIAgent.from_tools(
            [
                QueryEngineTool.from_defaults(
                    query_engine=vector_query_engine,
                    name="industry_agent",
                    description="這個工具提供所有公司的所屬相關產業別(industry)對照表。",
                )
            ],
            system_prompt=INDUSTRY_AGENT_PROMPT_EN,
            verbose=True,
        )

        # 建立 agent
        notes_agent = OpenAIAgent.from_tools(
            [
                QueryEngineTool.from_defaults(
                    query_engine=vector_query_engine,
                    name="notes_agent",
                    description="這個工具提供所有公司的備註資訊(notes)。",
                )
            ],
            system_prompt=NOTES_AGENT_PROMPT_EN,
            verbose=True,
        )

        return industry_agent, notes_agent

    async def build_esg_agent(self, esg_title: str):
        esg_path = os.path.join(self.esg_dir_path, esg_title)
        vector_index = await IndexBuilder.build_vector_index(
            f"{esg_path}/vector",
        )
        # summary_index = await IndexBuilder.build_doc_summary_index(
        #     f"{esg_path}/summary",
        # )
        vector_query_engine = vector_index.as_query_engine(
            similarity_top_k=10, use_async=True
        )
        # summary_query_engine = summary_index.as_query_engine(
        #     response_mode="simple_summarize",
        #     use_async=True,
        #     llm=OpenAI(temperature=0, model="gpt-3.5-turbo"),
        # )
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name=f"vector_tool_{esg_title.split('_')[0]}",
                    description=f"這份文件主要是{esg_title}在環境、社會及治理相關的資訊，傳達企業在永續經營上的規劃與成果，透過提高資訊透明度的方式，讓各個利害關係人能透過永續報告書，清楚的檢視企業的永續政策推動與管理成效。",
                ),
            ),
            # QueryEngineTool.from_defaults(
            #     query_engine=summary_query_engine,
            #     name=f"summary_tool_{esg_title.split('_')[0]}",
            #     description=f"提供關於{esg_title}的永續報告書的摘要、總結的工具",
            # ),
        ]
        openai_step_engine = OpenAIAgentWorker.from_tools(
            query_engine_tools, verbose=True, max_rollouts=3, num_expansions=2
        )
        return OpenAIAgent.from_tools(
            query_engine_tools,
            verbose=True,
            openai_step_engine=openai_step_engine,
        )

    async def build_esg_agents(
        self,
        esg_titles: list[str],
    ) -> dict[str, OpenAIAgent]:
        tasks = [self.build_esg_agent(esg_title) for esg_title in esg_titles]
        agents = await asyncio.gather(*tasks)
        return dict(zip(esg_titles, agents))
