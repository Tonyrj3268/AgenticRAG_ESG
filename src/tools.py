from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


class ToolManager:
    def __init__(self) -> None:
        self.tools = []

    def get_all_tools(self) -> list[QueryEngineTool]:
        return [tool for tool in self.tools]

    def add_industry_tool(self, industry_agent: OpenAIAgent) -> None:

        industry_description = """
        This content contains all the company's industry. Use this if you want to find the companies belong specific industry.
        """
        self.tools.append(
            QueryEngineTool(
                query_engine=industry_agent,
                metadata=ToolMetadata(
                    name=f"industry_tool",
                    description=industry_description,
                ),
            )
        )

    def add_note_tool(self, note_agent: OpenAIAgent) -> None:
        notes_description = """
        This content contains all the company's notes. 
        Always use this if you didn't see the notes been contained after the final esg_agent_tool be used.
        """

        self.tools.append(
            QueryEngineTool(
                query_engine=note_agent,
                metadata=ToolMetadata(
                    name="notes_tool",
                    description=notes_description,
                ),
            )
        )

    def add_document_tools(
        self, agents: dict[str, OpenAIAgent], esg_titles: list[str]
    ) -> None:
        all_tools = []
        for esg_title in esg_titles:
            doc_tool = QueryEngineTool(
                query_engine=agents[esg_title],
                metadata=ToolMetadata(
                    name=f"subAgent_{esg_title.split('_')[0]}",
                    description=f"ESG report about {esg_title}.",
                ),
            )
            all_tools.append(doc_tool)

        router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(), query_engine_tools=all_tools
        )

        self.tools.append(
            QueryEngineTool.from_defaults(
                query_engine=router_engine,
                name="esg_agent_tool",
                description="一個用於分析指定公司的ESG報告的工具。使用它来調查問題中指定的公司ESG報告中的環境、社會和治理實踐，比較永續發展倡議，或尋找有關企業責任和道德實踐的具體資訊。",
            )
        )
