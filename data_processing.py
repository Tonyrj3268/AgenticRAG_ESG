import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document


class DocumentLoader:
    """
    文檔加載器
    """

    @staticmethod
    def get_doc(file_path: str) -> list[Document]:
        return SimpleDirectoryReader(input_files=[file_path]).load_data()

    @staticmethod
    async def aget_doc(file_path: str) -> list[Document]:
        reader = await SimpleDirectoryReader(input_files=[file_path]).aload_data()
        return reader

    @staticmethod
    async def get_all_files_doc(file_paths: list[str]) -> list[list[Document]]:
        """
        並行處理多個文件。

        Args:
            file_paths (list[str]): 文件路徑列表

        Returns:
            list[list[Document]]: 所有文檔的數據列表
        """
        tasks = [DocumentLoader.aget_doc(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks)
