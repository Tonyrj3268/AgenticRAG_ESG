from llama_index.agent.openai import OpenAIAgent
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata


class ToolManager:
    def __init__(self) -> None:
        self.industry_tool = None
        self.note_tool = None
        self.esg_agent_tool = None

    def get_all_tools(self) -> list[QueryEngineTool]:
        return [
            tool
            for tool in (self.industry_tool, self.esg_agent_tool, self.note_tool)
            if tool
        ]

    def add_industry_tool(self, industry_agent: OpenAIAgent) -> None:

        industry_description = """
        This content contains all the company's industry. Use this if you want to find the companies belong specific industry.
        Always use compare_tool to get more information about the company.
        """
        self.industry_tool = QueryEngineTool(
            query_engine=industry_agent,
            metadata=ToolMetadata(
                name=f"industry_tool",
                description=industry_description,
            ),
        )

    def add_note_tool(self, note_agent: OpenAIAgent) -> None:
        notes_description = """
        This content contains all the company's notes. Use this if you want to find the companies belong specific notes.
        Always use this tool to get more information after using compare_tool.
        """
        self.note_tool = QueryEngineTool(
            query_engine=note_agent,
            metadata=ToolMetadata(
                name="notes_tool",
                description=notes_description,
            ),
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
                    description=f"This content contains ESG report about {esg_title}. Use this tool if you want to answer any questions about {esg_title}.",
                ),
            )
            all_tools.append(doc_tool)

        router_engine = RouterQueryEngine(
            selector=LLMSingleSelector.from_defaults(), query_engine_tools=all_tools
        )
        self.esg_agent_tool = QueryEngineTool.from_defaults(
            query_engine=router_engine,
            name="esg_agent_tool",
            description="一個用於分析指定公司的ESG報告的工具。使用它来調查問題中指定的公司ESG報告中的環境、社會和治理實踐，比較永續發展倡議，或尋找有關企業責任和道德實踐的具體資訊。",
        )
