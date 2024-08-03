import asyncio
import os

import pytest

from src.agents import AgentBuilder
from src.app import Config, SettingsManager


@pytest.mark.asyncio
async def test_build_esg_agent():
    # 設置測試環境
    SettingsManager.initialize()
    esg_dir_path = Config.ESG_DIR_PATH
    esg_title = "1217_愛之味"
    agent_builder = AgentBuilder(esg_dir_path)
    esg_agent = await agent_builder.build_esg_agent(esg_title)

    # 使用代理執行一個簡單的查詢
    query = "愛之味的獨立董事有誰?"
    response = esg_agent.query(query)

    print(f"Agent response: {response.response}")
    print(f"Source nodes: {response.source_nodes}")


if __name__ == "__main__":
    asyncio.run(test_build_esg_agent())
