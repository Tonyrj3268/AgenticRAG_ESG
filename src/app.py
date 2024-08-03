import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # 加載 .env 文件


class Config:
    LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
    project_root = Path(__file__).resolve().parent.parent
    ESG_DIR_PATH = os.path.join(project_root, os.getenv("ESG_DIR_PATH"))
    INDUSTRY_FILE_PATH = os.getenv(
        "INDUSTRY_FILE_PATH", os.path.join(ESG_DIR_PATH, "company_industry.json")
    )
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
from llama_index.agent.openai import OpenAIAgent
from llama_index.core import Settings
from llama_index.core.agent import ReActAgent
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from src.agents import AgentBuilder
from src.data_processing import DocumentLoader
from src.html_template import bot_template, css, user_template
from src.indexing import IndexBuilder
from src.prompts import (
    GENERAL_AGENT_PROMPT,
    GENERAL_AGENT_PROMPT_EN,
    GENERAL_AGENT_PROMPT_NO_NOTE_EN,
)
from src.tools import ToolManager


class SettingsManager:
    @staticmethod
    def initialize():
        model = "gpt-4o-mini"
        Settings.llm = OpenAI(temperature=0, model=model)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")


from typing import Optional

import llama_index.core.instrumentation as instrument
from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation.events import BaseEvent
from llama_index.core.instrumentation.span_handlers import SimpleSpanHandler


class NaiveEventHandler(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        """Class name."""
        return "NaiveEventHandler"

    @classmethod
    def _get_pretty_dict_str(
        cls, _dict: dict, skip_keys: Optional[list] = None, indent_str: str = ""
    ) -> str:
        _skip_keys = skip_keys or []

        ret = ""
        for _k, _v in _dict.items():
            if _k in _skip_keys:
                continue
            ret += f"{indent_str}{_k}: {_v}\n"
        return ret

    @classmethod
    def _get_pretty_even_str(cls, event: BaseEvent) -> str:
        _indent = "    "
        ret = ""

        for ek, ev in event.dict().items():
            if ek == "model_dict":
                continue
                # dict
                ret += f"{ek}:\n"
                ret += cls._get_pretty_dict_str(
                    ev, skip_keys=["api_key"], indent_str=_indent
                )
            elif ek == "embeddings":
                continue
                # List[List[float]]
                ret += f"{ek}: "
                ret += ",".join([f"<{len(_embedding)}-dim>" for _embedding in ev])
                ret += "\n"
            elif ek == "nodes":
                # List[NodeWithScore]
                # NodeWithScore is still too long; cannot think of a good repr in pure text
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            elif ek == "messages":
                # List[ChatMessage]
                ret += f"{ek}:\n"
                for _n in ev:
                    ret += f"{_indent}{_n}\n"
            else:
                ret += f"{ek}: {ev}\n"

        return ret

    def handle(self, event: BaseEvent, **kwargs):
        """Logic for handling event."""
        with open("log.txt", "a") as f:
            f.write(self._get_pretty_even_str(event))
            f.write("\n")


dispatcher = instrument.get_dispatcher()
dispatcher.add_event_handler(NaiveEventHandler())
span_handler = SimpleSpanHandler()
dispatcher.add_span_handler(span_handler)


class ESGAgent:
    def __init__(self) -> None:
        self.industry_doc = DocumentLoader(Config.LLAMAPARSE_API_KEY).get_doc(
            Config.INDUSTRY_FILE_PATH
        )
        self.agent_builder = AgentBuilder(Config.ESG_DIR_PATH)
        self.tool_manager = ToolManager()
        self.agent = None

    async def initialize(self) -> None:
        industry_note_agent_task = self.agent_builder.build_industry_note_agent(
            self.industry_doc
        )
        esg_agents_task = self.agent_builder.build_esg_agents(Config.list_companies())

        (industry_agent, _), esg_agents = await asyncio.gather(
            industry_note_agent_task, esg_agents_task
        )
        self.tool_manager.add_industry_tool(industry_agent)
        # self.tool_manager.add_note_tool(note_agent)
        self.tool_manager.add_document_tools(esg_agents, Config.list_companies())
        self.agent = OpenAIAgent.from_tools(
            tools=self.tool_manager.get_all_tools(),
            system_prompt=GENERAL_AGENT_PROMPT_NO_NOTE_EN,
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

    documents = await DocumentLoader(Config.LLAMAPARSE_API_KEY).get_all_files_doc(
        file_paths
    )
    tasks = [
        IndexBuilder().build_vector_index(
            f"{Config.ESG_DIR_PATH}/{os.path.splitext(os.path.basename(document[0].metadata['file_name']))[0]}/vector",
            data=document,
        )
        for document in documents
    ]

    await asyncio.gather(*tasks)
    st.session_state.companies = Config.list_companies()

    if "on_company_list_change" in st.session_state:
        st.session_state.on_company_list_change()

    # st.session_state.esg_agent = get_esg_agent()


def update_company_list() -> None:
    st.session_state.company_selector = st.session_state.companies


def update_sidebar_companies() -> None:
    st.sidebar.subheader("公司列表")
    if "companies" not in st.session_state:
        st.session_state.companies = Config.list_companies()
    st.session_state.on_company_list_change = update_company_list
    company_count = len(st.session_state.companies)
    st.sidebar.write(f"目前共有 {company_count} 家企業永續報告書")
    with st.sidebar.expander("點擊展開", expanded=False):
        if st.session_state.companies:
            for company in st.session_state.companies:
                st.write(company)
        else:
            st.write("目前沒有已存的企業永續報告書。")

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


def handle_userinput(user_question: str) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    response = st.session_state.esg_agent.query(user_question)

    st.session_state.chat_history.append(
        {"role": "assistant", "content": str(response)}
    )
    st.session_state.chat_history.append({"role": "user", "content": user_question})

    # Display chat history
    for message in st.session_state.chat_history[::-1]:
        template = user_template if message["role"] == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True
        )


# 主函數
def main():
    st.set_page_config(page_title="向你的PDF問問題", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.session_state.esg_agent = get_esg_agent()

    st.header("向多個PDF問問題 :books:")
    user_question = st.text_input(
        "問一個關於你文件的問題：（模型：GPT-4o · 生成的內容可能不准確或錯誤）"
    )
    if user_question:
        handle_userinput(user_question)
    update_sidebar_companies()


if __name__ == "__main__":
    nest_asyncio.apply()
    main()
