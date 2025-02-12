import sys
import chromadb
import os
import json
from langchain_openai import OpenAIEmbeddings
from util.remove_comments import remove_comments_and_docstrings


# os.environ["OPENAI_API_KEY"] = "your-api-key"  # 请替换为你的实际API密钥
# os.environ["OPENAI_BASE_URL"] = "your-base-url"  # 请替换为你的实际API密钥


# 对文本矢量化并存储在本地
def create_db():
    # 创建ChromaDB客户端
    client = chromadb.PersistentClient(path="./database")

    # 创建或获取名为"code_summary"的集合
    collection = client.get_or_create_collection(name="code_summary")

    # language = ['c', 'java', 'python']
    language = ['c']
    directory = "./dataset/Dataset"
    # load data
    for lang in language:
        # data_file = directory + '/' + lang + '.jsonl'
        data_file = directory + '/c.jsonl'
        codes = []
        comments = []
        length = 0
        with open(data_file, "r", encoding="utf-8") as f:
            print("opening file ", data_file)
            for line in f:
                length += 1
                line = line.strip()
                js = json.loads(line)
                if lang == 'c':
                    codes.append(remove_comments_and_docstrings(source=js['function'], lang=lang))
                    comment = js['summary']
                    if comment.endswith('.'):
                        comment = comment[:-1]
                    comment = comment + ' .'
                    comments.append(comment)
                else:
                    codes.append(remove_comments_and_docstrings(source=js['code'], lang=lang))
                    comments.append(' '.join(js['docstring_tokens']))

        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key="sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux",
            base_url="https://xiaoai.plus/v1"
        )
        code_embeddings = []
        # embedding_file_path = os.path.join('./', f"embedding_{lang}.jsonl")  # 文件名可以根据索引生成
        for idx, code in enumerate(codes):
            print(f"code {idx} is embedding......")
            embedding = embedding_function.embed_documents(code)
            # embedding 是一个二维列表（比如 [embedding1]）
            # with open(embedding_file_path, 'w') as f:
            #     f.write(str(embedding))
            code_embeddings.append(embedding[0])  # 这里使用 [0] 来获取一维嵌入
            print("=======embedding DONE=========")
        # summary_embeddings = [embedding_function.embed_documents(s) for s in comments]

        # 添加数据到集合
        collection.add(
            documents=codes,
            metadatas=[{"summary": s, "language": lang} for s in comments],
            embeddings=code_embeddings,
            ids=[str(i) for i in range(length)]
        )
        print(f"********* language {lang} saved,totally {length} ********")


def query_n(code, lang, n):
    # 查询与给定代码片段相似的摘要，并且只查找指定语言的代码
    embedding_function = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key="sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux",
        base_url="https://xiaoai.plus/v1"
    )

    query_code = code
    query_embedding = embedding_function.embed_query(query_code)

    # 创建或连接到已有的 Chroma 客户端
    client = chromadb.PersistentClient(path="./database")
    collection = client.get_or_create_collection(name="code_summary")

    # 执行相似度搜索，并加上过滤条件，只查找与指定语言匹配的代码
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n,  # 返回最相似的 n 个结果
        where={"language": lang}  # 过滤条件：只查询语言为指定 lang 的文档
    )
    # 提取并整理为所需的格式
    similar_samples = []
    # 遍历每一组 documents 和对应的 metadata
    for doc_list, metadata_list in zip(results['documents'], results['metadatas']):
        for doc, metadata in zip(doc_list, metadata_list):
            code = doc
            comment = metadata.get('summary', '')
            similar_samples.append({"code": code, "comment": comment})

    return similar_samples