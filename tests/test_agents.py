import asyncio
import os

import pytest

from src.agents import AgentBuilder
from src.app import Config


@pytest.mark.asyncio
async def test_build_esg_agent():
    # 設置測試環境
    esg_dir_path = Config.ESG_DIR_PATH
    esg_title = "5880_國庫金"

    agent_builder = AgentBuilder(esg_dir_path)
    esg_agent = await agent_builder.build_esg_agent(esg_title)

    # 使用代理執行一個簡單的查詢
    query = "請回答擁有合規風險有關的風險項目有什麼?"
    response = esg_agent.chat(query)

    print(f"Agent response: {response.response}")
    print(f"Source nodes: {response.source_nodes}")


if __name__ == "__main__":
    asyncio.run(test_build_esg_agent())
