import nest_asyncio

from src.app import main

if __name__ == "__main__":
    nest_asyncio.apply()
    main()
