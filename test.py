import asyncio

from app import AgentBuilder, Config, DocumentLoader, SettingsManager


async def test_industry_agent_builder():
    agent_builder = AgentBuilder(Config.BASE_DIR, Config.ESG_DIR_PATH)
    industry_doc = DocumentLoader.get_doc(Config.INDUSTRY_FILE_PATH)

    (industry_agent, note_agent) = await agent_builder.build_industry_note_agent(
        industry_doc
    )
    print(industry_agent.query("食品業"))


async def test_note_agent_builder():
    agent_builder = AgentBuilder(Config.BASE_DIR, Config.ESG_DIR_PATH)
    industry_doc = DocumentLoader.get_doc(Config.INDUSTRY_FILE_PATH)

    (industry_agent, note_agent) = await agent_builder.build_industry_note_agent(
        industry_doc
    )
    print(note_agent.query("長榮"))


if __name__ == "__main__":
    SettingsManager.initialize()
    asyncio.run(test_note_agent_builder())
