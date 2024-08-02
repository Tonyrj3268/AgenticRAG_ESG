import asyncio
import cProfile
import io
import pstats
from functools import wraps

from llama_index.core import VectorStoreIndex, load_index_from_storage
from llama_index.core.storage import StorageContext

from src.agents import AgentBuilder
from src.app import Config


def async_profile(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        result = asyncio.run(func(*args, **kwargs))
        pr.disable()
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).strip_dirs().sort_stats("tottime")
        ps.print_stats(100)
        return result, s.getvalue()

    return wrapper


async def test_build_esg_agent():
    agent_builder = AgentBuilder(Config.ESG_DIR_PATH)
    agent = await agent_builder.build_esg_agent("5880_合庫金")
    print(agent.query("請回答氣候相關風險與機會對組織在業務、策略和財務規劃的影響？"))


@async_profile
async def test_load_index_from_storage():
    load_index_from_storage(
        StorageContext.from_defaults(
            persist_dir=f"{Config.ESG_DIR_PATH}/5880_合庫金/vector"
        )
    )


if __name__ == "__main__":
    # _,results = test_build_esg_agent()
    # print(results)
    asyncio.run(test_build_esg_agent())
