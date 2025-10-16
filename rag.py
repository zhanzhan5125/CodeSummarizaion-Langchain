import os

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
import json
from Result.info import low_score
from util.remove_comments import remove_comments_and_docstrings


# ========================== 数据准备与数据库构建 ==========================
def create_db_langchain(language):
    """ 使用 LangChain 的 Chroma 集成构建数据库 """
    # 初始化嵌入模型
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-large",
        # api_key='sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
        # base_url="https://api.agicto.cn/v1",
        api_key='sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7',
        base_url="https://api.mjdjourney.cn/v1",
    )
    # 使用 LangChain 的 Chroma 集成自动处理嵌入和存储
    vector_db = Chroma(
        persist_directory="./database",  # 指定持久化目录
        collection_name=language,
        embedding_function=embedding_model
    )
    print("Database created with LangChain integration!")
    return vector_db


# ========================== 添加数据 ==========================
def add_data(data_file, lang, ignore):
    # 加载数据并转换为 LangChain 的 Document 格式
    vector_db = create_db_langchain(lang)
    documents = []

    # language = ['java', 'python']
    language = ['java']

    for lang in language:
        data_dir = f"./Dataset/{lang}/train"
        data_file_path = os.path.join(data_dir, data_file)
        with open(data_file_path, "r", encoding="utf-8") as f:
            print(f"Loading data from {data_file}")
            for idx, line in enumerate(f, start=1):
                if idx in ignore: continue
                js = json.loads(line.strip())
                code = remove_comments_and_docstrings(source=js['code'], lang=lang)
                comment = ' '.join(js['docstring_tokens'])
                data = {'code': code, 'comment': comment, 'repo': js['repo'], 'path': js['path'],
                        'func': js['func_name']}
                # 构建 LangChain Document 对象
                doc = Document(
                    page_content=code,  # 代码作为主要内容
                    metadata={
                        "data": json.dumps(data)
                    }
                )
                documents.append(doc)

    print("Starting embedding")
    vector_db.add_documents(documents=documents)
    print("Finished embedding")


# ========================== 删除某数据模块 ==========================
def delete_by_content_and_metadata(del_list, lang):
    # 初始化 Chroma 实例（复用已有数据库配置）
    chroma_db = create_db_langchain(lang)
    # 获取集合中所有文档ID
    existing_data = chroma_db.get()  # 返回包含 ids/documents/metadatas 的字典
    matching_ids = []
    print(len(existing_data))
    for doc_id, meta in zip(existing_data["ids"], existing_data["metadatas"]):
        data = json.loads(meta["data"])
        if data["code"] in del_list:
            matching_ids.append(doc_id)

    print(f"Deleting {len(matching_ids)} documents")
    if matching_ids:
        chroma_db.delete(ids=matching_ids)


def chroma_size(lang):
    # 初始化 Chroma 实例（复用已有数据库配置）
    chroma_db = create_db_langchain(lang)
    # 获取底层 Chroma 客户端和集合
    collection = chroma_db._client.get_collection(lang)
    print(f"Still exist {collection.count()} documents")
    print(json.loads(chroma_db.get()["metadatas"][0]['data']))


def build_bm25(n, lang):
    chroma_db = create_db_langchain(lang)
    existing_data = chroma_db.get()
    documents = []
    for content, meta in zip(existing_data["documents"], existing_data["metadatas"]):
        documents.append(Document(page_content=content, metadata=meta))
    bm25 = BM25Retriever.from_documents(documents, k=n)
    return bm25


# ========================== 查询模块 ==========================
class CodeRetriever:
    def __init__(self, k, lang):
        # 初始化数据库连接
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            # api_key='sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
            # base_url="https://api.agicto.cn/v1",
            api_key='sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7',
            base_url="https://api.mjdjourney.cn/v1",
            # timeout=30
        )

        self.vector_db = Chroma(
            persist_directory="./database",
            embedding_function=self.embedding_model,
            collection_name=lang
        )

        # 配置检索器（带过滤条件）
        self.embedding = self.vector_db.as_retriever(
            search_kwargs={
                "k": k
            }
        )
        self.bm25 = build_bm25(k, lang)

    def query_samples(self, data, method):
        docs = []
        code = data['code']
        # 动态更新过滤条件
        # 执行检索
        if method == 'embedding':
            docs = self.embedding.invoke(code)
        elif method == 'bm25':
            docs = self.bm25.invoke(code)
        # 格式化为统一输出
        if len(docs) == 0:
            print("No documents found")

        examples = [
            {
                "code": doc.page_content,
                "data": json.loads(doc.metadata["data"])
            } for doc in docs
        ]
        return examples


if __name__ == "__main__":
    # create_db_langchain('java')
    # low = low_score("./Result/trainset_validation/10000_java_valid.csv", 1)
    # print(len(low))
    # add_data("2000_java_train.jsonl", 'java', [])
    # testfile = "./Dataset/python/train/10000_python_train.jsonl"
    # with open(testfile, "r", encoding="utf-8") as f:
    #     print("opening file ", testfile)
    #     cnt = 0
    #     for line in f:
    #         cnt += 1
    #         if cnt < 0: continue
    #         line = line.strip()
    #         js = json.loads(line)
    #         code = remove_comments_and_docstrings(source=js['code'], lang='python')
    #         if cnt in low:
    #             low_code.append(code)
    #     print(len(low_code))
    # delete_by_content_and_metadata(low_code, 'python')
    print(chroma_size('java'))
    # add_data('1000_python_train.jsonl', 'python',ignore=[])

