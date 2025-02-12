import os
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from util.remove_comments import remove_comments_and_docstrings
import json


# ========================== 数据准备与数据库构建 ==========================
def create_db_langchain():
    """ 使用 LangChain 的 Chroma 集成构建数据库 """
    # 初始化嵌入模型
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key="sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux",
        base_url="https://xiaoai.plus/v1"
    )

    # 加载数据并转换为 LangChain 的 Document 格式
    documents = []
    metadatas = []

    language = ['c', 'java', 'python']
    # language = ['c']
    directory = "./dataset/Dataset"

    for lang in language:
        data_file = directory + '/' + lang + '.jsonl'
        with open(data_file, "r", encoding="utf-8") as f:
            print(f"Loading data from {data_file}")
            for line in f:
                js = json.loads(line.strip())

                # 预处理代码（移除注释）
                raw_code = js['function'] if lang == 'c' else js['code']
                processed_code = remove_comments_and_docstrings(source=raw_code, lang=lang)

                # 处理注释
                comment = js['summary'] if lang == 'c' else ' '.join(js['docstring_tokens'])
                if lang == 'c' and comment.endswith('.'):
                    comment = comment[:-1] + ' .'  # 保持你的原始处理逻辑

                # 构建 LangChain Document 对象
                doc = Document(
                    page_content=processed_code,  # 代码作为主要内容
                    metadata={
                        "summary": comment,
                        "language": lang
                    }
                )
                documents.append(doc)

    # 使用 LangChain 的 Chroma 集成自动处理嵌入和存储
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory="./database",  # 指定持久化目录
        collection_name="code_summary"
    )
    print("Database created with LangChain integration!")


# ========================== 查询模块 ==========================
class CodeRetriever:
    def __init__(self, k):
        # 初始化数据库连接
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key="sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux",
            base_url="https://xiaoai.plus/v1"
        )

        self.vector_db = Chroma(
            persist_directory="./database",
            embedding_function=self.embedding_model,
            collection_name="code_summary"
        )

        # 配置检索器（带过滤条件）
        self.retriever = self.vector_db.as_retriever(
            search_kwargs={
                "k": k,
                "filter": {"language": "c"}  # 示例默认过滤条件
            }
        )

    def query_samples(self, code: str, lang: str = "c", n: int = 3) -> list[dict]:
        # 动态更新过滤条件
        self.retriever.search_kwargs["filter"]["language"] = lang
        self.retriever.search_kwargs["k"] = n

        # 执行检索
        docs = self.retriever.invoke(code)

        print(docs)

        # 格式化为统一输出
        return [
            {
                "code": doc.page_content,
                "comment": doc.metadata["summary"]
            } for doc in docs
        ]


# ========================== 使用示例 ==========================
if __name__ == "__main__":
    # 重建数据库（只需运行一次）
    # create_db_langchain()

    # 初始化检索器
    retriever = CodeRetriever(k=3)

    # 测试查询
    test_code = "def network_up"
    results = retriever.query_samples(test_code, lang="python", n=2)
    print("Retrieved samples:", results)
