import asyncio

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document
from llama_parse import LlamaParse


class DocumentLoader:
    """文檔加載器類，提供多種方法來加載和處理文檔。"""

    def __init__(self, llamaparse_api_key: str) -> None:
        """
        初始化DocumentLoader。

        Args:
            llamaparse_api_key (str): LlamaParse API密鑰
        """
        self.json_parser = LlamaParse(
            api_key=llamaparse_api_key,
            result_type="json",
            num_workers=4,
            verbose=True,
            language="ch_tra",
        )
        self.md_parser = LlamaParse(
            api_key=llamaparse_api_key,
            result_type="markdown",
            num_workers=4,
            verbose=True,
            language="ch_tra",
        )

    def get_doc(self, file_path: str) -> list[Document]:
        """
        同步加載單個PDF文件。

        Args:
            file_path (str): PDF文件的路徑

        Returns:
            list[Document]: 包含文檔內容的Document對象列表
        """
        file_extractor = {".pdf": self.md_parser}
        document = SimpleDirectoryReader(
            file_extractor=file_extractor, input_files=[file_path]
        ).load_data()
        for i, doc in enumerate(document):
            doc.metadata = {"pages": i + 1, "file_name": file_path.split("/")[-1]}
            doc.text_template = str(
                "Metadata: {metadata_str}\n-----\nContent: {content}"
            )
        return document

    async def aget_doc(self, file_path: str) -> list[Document]:
        """
        異步加載單個文件。

        Args:
            file_path (str): 文件路徑

        Returns:
            list[Document]: 包含文檔內容的Document對象列表
        """
        document = await self.md_parser.aload_data(file_path)
        for i, doc in enumerate(document):
            doc.metadata = {"pages": i + 1, "file_name": file_path.split("/")[-1]}
            doc.text_template = str(
                "Metadata: {metadata_str}\n-----\nContent: {content}"
            )
        return document

    async def get_json_doc(self, file_path: str) -> list[Document]:
        """
        使用LlamaParse解析文件並處理返回的JSON格式數據。

        Args:
            file_path (str): 文件路徑

        Returns:
            list[Document]: 包含文檔內容的Document對象列表
        """

        json_objs = await self.json_parser.aget_json(file_path)
        json_list = json_objs[0]["pages"]
        documents = []
        for page in json_list:
            documents.append(
                Document(
                    text=page.get("text"),
                    metadata={
                        "pages": page.get("page"),
                        "file_name": file_path.split("/")[-1],
                    },
                    text_template="Metadata: {metadata_str}\n-----\nContent: {content}",
                )
            )
        return documents

    async def get_all_files_doc(self, file_paths: list[str]) -> list[list[Document]]:
        """
        並行處理多個文件。

        Args:
            file_paths (list[str]): 文件路徑列表

        Returns:
            list[list[Document]]: 所有文檔的數據列表
        """

        tasks = [self.get_json_doc(file_path) for file_path in file_paths]
        return await asyncio.gather(*tasks)
