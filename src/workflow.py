from dotenv import load_dotenv

load_dotenv()  # 加載 .env 文件
import asyncio
import json
import logging
import os
from pathlib import Path

import streamlit as st
from llama_index.agent.openai import OpenAIAgent
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core.settings import Settings
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI

from agents import AgentBuilder
from events import *
from html_template import bot_template, css, user_template
from prompts import GENERATION_PROMPT, REFLECTION_PROMPT


def setup_logging():
    logging.basicConfig(
        filename=Config.LOG_FILE_PATH,
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_event(event_type: str, data: dict):
    log_entry = f"{event_type}: {json.dumps(data, ensure_ascii=False)}"
    logging.info(log_entry + "\n" + "-" * 50)


class SettingsManager:
    @staticmethod
    def initialize():
        model = "gpt-4o-mini"
        Settings.llm = OpenAI(temperature=0, model=model)
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-large")
        llama_debug = LlamaDebugHandler(print_trace_on_end=True)
        Settings.callback_manager = CallbackManager([llama_debug])


class Config:
    LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")
    project_root = Path(__file__).resolve().parent.parent
    ESG_DIR_PATH = os.path.join(project_root, os.getenv("ESG_DIR_PATH"))
    INDUSTRY_FILE_PATH = os.path.join(project_root, os.getenv("INDUSTRY_FILE_PATH"))
    VERBOSE = os.getenv("VERBOSE", "True").lower() == "true"
    LOG_FILE_PATH = os.path.join(project_root, "chat_logs.txt")

    @classmethod
    def list_companies(cls) -> list[str]:
        return [
            dirname
            for dirname in os.listdir(cls.ESG_DIR_PATH)
            if os.path.isdir(os.path.join(cls.ESG_DIR_PATH, dirname))
            and os.path.exists(os.path.join(cls.ESG_DIR_PATH, dirname, "vector"))
            and dirname != "industry_map"
        ]


class QuerySplitWorkflow(Workflow):
    def __init__(
        self,
        esg_agents_map: dict[str, OpenAIAgent],
        industry_map: list[IndustryMap],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ai_model = OpenAI(temperature=0, model="gpt-4o-mini")
        self.esg_agents_map = esg_agents_map
        self.industry_map = industry_map

    @step(pass_context=True)
    async def receive_query(
        self, ctx: Context, ev: StartEvent
    ) -> QueryReceivedEvent | StopEvent:
        main_query = ev.get("question")
        if not main_query:
            return StopEvent(result="Please provide a main query")
        ctx.data["main_query"] = main_query
        return QueryReceivedEvent(query=main_query)

    @step()
    async def determine_query_type(
        self, ev: QueryReceivedEvent
    ) -> ProcessedQueryEvent | StopEvent:
        main_query = ev.query
        industry_prompt = [
            f"{item.company} 是 {str(item.industry)} 的公司，其別名是 {str(item.alias)}"
            for item in self.industry_map
        ]
        prompt = f"""
        請判斷以下主問題提及的產業或公司是否存在於產業查詢表中：{main_query} /n
        以下是產業的查詢表：{industry_prompt} /n
        如果不是產業相關查詢，請直接返回主問題，不要有任何其他的說明。 /n
        如果是產業相關查詢，請將產業的部分換成相關的公司名稱，並返回新的主問題。 /n
        有部分公司可能有別名(alias)，例如合庫金也叫合庫金控，請仔細確認。
        如果該公司或是產業不存在於產業查詢表中，請返回「不存在」，不要有任何其他的說明。 /n
        例如
        主問題：請分析食品業的薪水？ /n
        根據產業查詢表，產業是食品業，將產業的部分換成相關的公司名稱，其中包括愛之味和台榮公司，並返回新的主問題。 /n
        新的主問題：請分析愛之味和台榮公司的薪水？ /n
        """
        response = await self.ai_model.acomplete(prompt)
        if "不存在" in str(response):
            return StopEvent(result="該公司或是產業不存在於產業查詢表中")
        return ProcessedQueryEvent(query=str(response))

    @step()
    async def split_query(
        self, ev: ProcessedQueryEvent | QuerySplitErrorEvent
    ) -> QuerySplitOutputEvent:

        if isinstance(ev, ProcessedQueryEvent):
            main_query = ev.query
            reflection_prompt = ""
        elif isinstance(ev, QuerySplitErrorEvent):
            main_query = ev.main_query
            reflection_prompt = REFLECTION_PROMPT.format(
                error=ev.error, main_query=main_query
            )
        prompt = GENERATION_PROMPT.format(
            main_question=main_query, schema=Subquery.model_json_schema()
        )
        if reflection_prompt:
            prompt += reflection_prompt
        response = await self.ai_model.acomplete(prompt)
        return QuerySplitOutputEvent(output=str(response), original_query=main_query)

    @step()
    async def validate_split(
        self, ev: QuerySplitOutputEvent
    ) -> SubqueriesGeneratedEvent | QuerySplitErrorEvent:
        try:
            output = json.loads(ev.output)
            if isinstance(output, list):
                return SubqueriesGeneratedEvent(
                    subqueries=[Subquery(**item) for item in output]
                )
            return SubqueriesGeneratedEvent(subqueries=[Subquery(**output)])
        except Exception as e:
            print("Output:", ev.output)
            return QuerySplitErrorEvent(
                error=str(e), invalid_output=ev.output, main_query=ev.original_query
            )

    @step(pass_context=True)
    async def prepare_subqueries(
        self, ctx: Context, ev: SubqueriesGeneratedEvent
    ) -> ChooseAgentEvent | None:
        subqueries = ev.subqueries
        ctx.data["subqueries_count"] = len(subqueries)
        for subquery in subqueries:
            self.send_event(ChooseAgentEvent(query=subquery.query))
        return None

    @step(num_workers=3)
    async def choose_esg_agent(self, ev: ChooseAgentEvent) -> RetrieverEvent:
        query = ev.query
        prompt = f"""
        請選擇一個ESG公司代理人來回答以下問題：{query}，
        選擇的公司代理人必須是以下列表中的一個：{list(self.esg_agents_map.keys())}，
        請直接回答選擇的公司代理人名稱，
        不要說'選擇的公司代理人是'等語句"""
        response = await self.ai_model.acomplete(prompt)
        agent = self.esg_agents_map[str(response)]
        return RetrieverEvent(agent=agent, query=query)

    @step()
    async def retrieve(self, ev: RetrieverEvent) -> RetrieverResponseEvent:
        agent: OpenAIAgent = ev.agent
        query = ev.query
        response = await agent.aquery(query)
        return RetrieverResponseEvent(response=str(response))

    @step(pass_context=True)
    async def collect_ai_responses(
        self, ctx: Context, ev: RetrieverResponseEvent
    ) -> StopEvent | None:
        result = ctx.collect_events(
            ev, [RetrieverResponseEvent] * ctx.data["subqueries_count"]
        )
        if result is None:
            return None
        main_query = ctx.data["main_query"]
        answer = str(result)
        prompt = f"""
        You are a professional ESG analyst. Based on the following main query and collected responses, provide a comprehensive and insightful answer:

        Main Query: {main_query}

        Collected Responses:
        {answer}

        Please synthesize the information and provide a well-structured, professional response that addresses the main query. Ensure your answer is clear, concise, and tailored to an audience interested in ESG matters.
        Always remember give the answer with the data source.
        Always return the answer in Traditional Chinese.
        """
        response = await self.ai_model.acomplete(prompt)
        return StopEvent(result=str(response))


def update_sidebar_companies() -> None:
    st.sidebar.subheader("公司列表")
    if "companies" not in st.session_state:
        st.session_state.companies = Config.list_companies()
    company_count = len(st.session_state.companies)
    st.sidebar.write(f"目前共有 {company_count} 家企業永續報告書")
    with st.sidebar.expander("點擊展開", expanded=False):
        if st.session_state.companies:
            for company in st.session_state.companies:
                st.write(company)
        else:
            st.write("目前沒有已存的企業永續報告書。")


def handle_userinput(user_question: str) -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    log_event("USER_INPUT", {"question": user_question})

    workflow = QuerySplitWorkflow(
        timeout=300,
        verbose=True,
        esg_agents_map=st.session_state.esg_agents_map,
        industry_map=st.session_state.industry_map,
    )

    response = asyncio.run(workflow.run(question=user_question))
    log_event("AI_RESPONSE", {"response": str(response)})

    st.session_state.chat_history.append(
        {"role": "assistant", "content": str(response)}
    )
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    for message in st.session_state.chat_history[::-1]:
        template = user_template if message["role"] == "user" else bot_template
        st.write(
            template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True
        )


async def initialize_agents():
    SettingsManager.initialize()
    agent_builder = AgentBuilder(Config.ESG_DIR_PATH)
    return await agent_builder.build_esg_agents(Config.list_companies())


def main():
    st.set_page_config(page_title="向你的PDF問問題", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)
    st.header("向多個PDF問問題 :books:")
    setup_logging()
    if "esg_agents_map" not in st.session_state:
        st.session_state.esg_agents_map = asyncio.run(initialize_agents())
    if "industry_map" not in st.session_state:
        with open(Config.INDUSTRY_FILE_PATH, "r") as f:
            industry_data = json.load(f)
            st.session_state.industry_map = [
                IndustryMap(
                    company=item["company"].split("_")[1],
                    industry=item["industry"],
                    alias=item["alias"],
                )
                for item in industry_data
            ]
    user_question = st.text_input(
        "問一個關於你文件的問題：（模型：GPT-4o-mini · 生成的內容可能不准確或錯誤）"
    )

    if user_question:
        handle_userinput(user_question)
    update_sidebar_companies()


if __name__ == "__main__":
    # draw_all_possible_flows(QuerySplitWorkflow, filename="ESG_workflow.html")
    main()
