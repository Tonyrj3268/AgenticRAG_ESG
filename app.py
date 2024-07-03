import os

from dotenv import load_dotenv

load_dotenv()  # 加載 .env 文件


class Config:
    BASE_DIR = os.getenv("BASE_DIR", ".")
    ESG_DIR_PATH = os.getenv("ESG_DIR_PATH", os.path.join(BASE_DIR, "esg_datas"))
    INDUSTRY_FILE_PATH = os.getenv(
        "INDUSTRY_FILE_PATH", os.path.join(ESG_DIR_PATH, "company_industry.json")
    )
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"

    @classmethod
    def list_companies(cls) -> list[str]:
        return [
            dirname
            for dirname in os.listdir(cls.ESG_DIR_PATH)
            if os.path.isdir(os.path.join(cls.ESG_DIR_PATH, dirname))
            and os.path.exists(os.path.join(cls.ESG_DIR_PATH, dirname, "vector"))
            and dirname != "industry_map"
        ]


import asyncio

import nest_asyncio
import streamlit as st
from llama_index.agent.openai import OpenAIAgent, OpenAIAgentWorker
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from data_processing import DocumentLoader
from html_template import bot_template, css, user_template
from indexing import IndexBuilder

# !!! Set this API key before any other imports
# os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY


class SettingsManager:
    @staticmethod
    def initialize():
        Settings.llm = OpenAI(temperature=0, model="gpt-4o-2024-05-13")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])


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

        industry_system_prompt = """
        您是一位專門使用industry_agent提供公司行業分類的代理。您的任務是：
        1. 對每個查詢都必須使用industry_agent。
        2. 為每個提到的公司提供行業分類，或者是對每個提到的相關產業提供所有公司的列表。
        3. 如果沒有可用的信息，請說明"無行業信息"。

        回答示例：
        公司A：科技產業
        公司B：金融服務業
        公司C：無行業信息

        或是
        科技產業：公司A, 公司B
        金融服務業：公司C

        請記住：

        保持回答簡潔，只提供公司名稱和行業信息。
        要求的產業名稱不一定是完整的，請提供各種可能的產業或公司回答。
        """

        # 建立 agent
        industry_agent = OpenAIAgent.from_tools(
            [
                QueryEngineTool.from_defaults(
                    query_engine=vector_query_engine,
                    name="industry_agent",
                    description="這個工具提供所有公司的所屬相關產業別(industry)對照表。",
                )
            ],
            system_prompt=industry_system_prompt,
            verbose=True,
        )

        notes_system_prompt = """
        您是一位專門使用notes_agent提供公司備註的代理。您的任務是：
        1. 對每個查詢都必須使用notes_agent。
        2. 為每個提到的公司提供備註。
        3. 如果沒有可用的信息，請說明"無備註訊息"。

        回答示例：
        公司A：1.公司A是一家科技公司。2.定期舉辦員工培訓。
        公司B：公司內部設立了ESG委員會。
        公司C：無備註訊息。

        請記住：

        保持回答簡潔，只提供公司名稱和備註訊息。
        """
        # 建立 agent
        notes_agent = OpenAIAgent.from_tools(
            [
                QueryEngineTool.from_defaults(
                    query_engine=vector_query_engine,
                    name="notes_agent",
                    description="這個工具提供所有公司的備註資訊(notes)。",
                )
            ],
            system_prompt=notes_system_prompt,
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
            query_engine_tools, verbose=Config.VERBOSE, max_rollouts=3, num_expansions=2
        )
        return OpenAIAgent.from_tools(
            query_engine_tools,
            verbose=Config.VERBOSE,
            openai_step_engine=openai_step_engine,
        )

    async def build_esg_agents(
        self,
        esg_titles: list[str],
        esg_docs={},
    ) -> dict[str, OpenAIAgent]:
        tasks = [self.build_esg_agent(esg_title) for esg_title in esg_titles]
        agents = await asyncio.gather(*tasks)
        return dict(zip(esg_titles, agents))


class ToolManager:
    def __init__(self):
        self.industry_tool = None
        self.note_tool = None
        self.esg_agent_tool = None

    def get_all_tools(self):
        return [
            tool
            for tool in (self.industry_tool, self.esg_agent_tool, self.note_tool)
            if tool
        ]

    def add_industry_tool(self, industry_agent):

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

    def add_note_tool(self, note_agent):
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

    def add_document_tools(self, agents: dict[str, OpenAIAgent], esg_titles: list[str]):
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


class ESGAgent:
    def __init__(self):
        self.industry_doc = DocumentLoader.get_doc(Config.INDUSTRY_FILE_PATH)
        self.agent_builder = AgentBuilder(Config.BASE_DIR, Config.ESG_DIR_PATH)
        self.tool_manager = ToolManager()
        self.agent = None

    async def initialize(self):
        industry_note_agent_task = self.agent_builder.build_industry_note_agent(
            self.industry_doc
        )
        esg_agents_task = self.agent_builder.build_esg_agents(Config.list_companies())

        (industry_agent, note_agent), esg_agents = await asyncio.gather(
            industry_note_agent_task, esg_agents_task
        )
        self.tool_manager.add_industry_tool(industry_agent)
        self.tool_manager.add_note_tool(note_agent)
        self.tool_manager.add_document_tools(esg_agents, Config.list_companies())
        self.agent = OpenAIAgent.from_tools(
            tools=self.tool_manager.get_all_tools(),
            system_prompt="""
            您是一個先進的代理人，您的手上有三個工具：industry_tool、esg_agent_tool 和 notes_tool。
            
            處理流程

                1. 分析用戶查詢,確定是針對特定公司還是整個行業。
                2. 如果查詢提到具體公司:
                    a. 直接進行步驟 4。
                3. 如果查詢提到行業而非具體公司:
                    a. 使用 industry_tool 獲取該行業的所有相關公司。
                    b. 對每家識別出的公司執行步驟4 。
                4. 使用 esg_agent_tool 針對每家公司查詢相關的 ESG 信息。
                5. 使用 notes_tool 獲取每家公司的額外註釋，但只保留與查詢直接相關的信息。
                6. 整合所有收集到的信息,提供全面的回答。

            回答格式
            對每家相關公司,提供以下信息:

            公司名稱:[名稱]
            行業:[行業分類]
            ESG 相關信息:[針對查詢的具體 ESG 資訊]
            備註:[僅包含與查詢直接相關的備註，如無則省略此項]

            問題範例及處理流程

            1. 愛之味的員工薪水是多少？

            處理流程:
            a. 識別這是針對單一公司(愛之味)的查詢
            b. 使用 industry_tool 確認愛之味的行業
            c. 使用 esg_agent_tool 查詢 "愛之味的員工薪水"
            d. 使用 notes_tool 獲取愛之味的備註，但僅保留與員工薪水相關的信息
            e. 整合信息並回答

            2. 請分析食品業的員工薪水

            處理流程:
            a. 識別這是針對整個行業(食品業)的查詢
            b. 使用 industry_tool 查詢食品業的所有公司(例如:愛之味、統一)
            c. 對每家公司:

            使用 esg_agent_tool 查詢 "請分析[公司名稱]的員工薪水"
            使用 notes_tool 獲取公司備註，但僅保留與員工薪水相關的信息
            d. 整合所有公司的信息,分析食品業整體薪水情況
            """,
            verbose=Config.VERBOSE,
        )


@st.cache_resource
def get_esg_agent():
    SettingsManager.initialize()
    esg_agent = ESGAgent()
    asyncio.run(esg_agent.initialize())
    return esg_agent.agent


async def process_documents(pdf_docs):
    file_paths = []
    for pdf_doc in pdf_docs:
        company_name = os.path.splitext(pdf_doc.name)[0]
        company_dir = os.path.join(Config.ESG_DIR_PATH, company_name)
        os.makedirs(company_dir, exist_ok=True)
        pdf_path = os.path.join(company_dir, pdf_doc.name)
        with open(pdf_path, "wb") as f:
            f.write(pdf_doc.getbuffer())
        file_paths.append(pdf_path)

    documents = await DocumentLoader.get_all_files_doc(file_paths)

    tasks = [
        IndexBuilder.build_vector_index(
            f"{Config.ESG_DIR_PATH}/{os.path.splitext(os.path.basename(doc[0].metadata['file_name']))[0]}/vector",
            doc=doc,
        )
        for doc in documents
    ]
    tasks.extend(
        [
            IndexBuilder.build_doc_summary_index(
                f"{Config.ESG_DIR_PATH}/{os.path.splitext(os.path.basename(doc[0].metadata['file_name']))[0]}/summary",
                doc=doc,
            )
            for doc in documents
        ]
    )

    await asyncio.gather(*tasks)
    st.session_state.companies = Config.list_companies()

    if "on_company_list_change" in st.session_state:
        st.session_state.on_company_list_change()


def update_company_list():
    st.session_state.company_selector = st.session_state.companies


def update_sidebar_companies():
    st.sidebar.subheader("公司列表")
    if "companies" not in st.session_state:
        st.session_state.companies = Config.list_companies()
    st.session_state.on_company_list_change = update_company_list
    with st.sidebar.expander("點擊展開", expanded=False):
        if st.session_state.companies:
            for company in st.session_state.companies:
                st.write(company)
        else:
            st.write("目前沒有已存的公司。")

    with st.sidebar:
        st.subheader("你的文件")
        pdf_docs = st.file_uploader(
            "在這裡上傳你的PDF並點擊'處理'", accept_multiple_files=True
        )
        if st.button("處理"):
            with st.spinner("處理中"):
                if pdf_docs:
                    asyncio.run(process_documents(pdf_docs))
                    st.success("文件處理完成！")
                else:
                    st.write("沒有文件被上傳。")


def handle_userinput(user_question):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    response = st.session_state.esg_agent.query(user_question)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append(
        {"role": "assistant", "content": str(response)}
    )

    # Display chat history
    for message in st.session_state.chat_history:
        template = user_template if message["role"] == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True
        )


# 主函數
def main():
    st.set_page_config(page_title="與你的PDF聊天", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.session_state.esg_agent = get_esg_agent()

    st.header("與多個PDF聊天 :books:")
    user_question = st.text_input(
        "問一個關於你文件的問題：（模型：GPT-4o · 生成的內容可能不准確或錯誤）"
    )
    if user_question:
        handle_userinput(user_question)
    update_sidebar_companies()


if __name__ == "__main__":
    nest_asyncio.apply()
    main()
