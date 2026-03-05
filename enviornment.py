import json
import numpy as np
from langchain_openai import OpenAIEmbeddings
from memory import Trajectory
# 从你现有的 rag.py 中导入数据库连接函数
from rag import create_db_langchain


class DistributionObserver:
    """
    外部环境：利用 Chroma 数据库中存储的摘要计算 Concise Cluster 的质心，
    生成当前生成的摘要草稿的 Observation。
    """

    def __init__(self, lang: str):
        self.lang = lang
        # 保持与 rag.py 一致的 Embedding 配置
        self.embedding_model = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key='sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7',
            base_url="https://api.mjdjourney.cn/v1",
        )
        # 初始化时，直接从 Chroma 数据库提取摘要并计算质心
        self.centroid = self._calculate_centroid_from_chroma()

    def _calculate_centroid_from_chroma(self) -> np.ndarray:
        """
        从 Chroma 中提取所有 metadata 里的 comment，并计算它们的中心向量 (Centroid)
        """
        # 1. 连接已有数据库
        vector_db = create_db_langchain(self.lang)

        # 2. 获取数据库中所有的记录
        existing_data = vector_db.get()
        metadatas = existing_data.get("metadatas", [])

        summaries = []
        for meta in metadatas:
            if "data" in meta:
                # 反序列化 metadata 获取 comment
                data_dict = json.loads(meta["data"])
                comment = data_dict.get("comment", "").strip()
                if comment:
                    summaries.append(comment)

        if not summaries:
            raise ValueError(f"Chroma 数据库中未找到 {self.lang} 相关的摘要数据！请先运行 rag.py 存入数据。")

        print(f"[{self.lang}] 正在为 Chroma 中的 {len(summaries)} 条摘要计算质心 (Centroid)...")

        # 3. 批量获取 embedding (分批次处理，防止 API 触碰限流)
        batch_size = 500
        all_embeddings = []
        for i in range(0, len(summaries), batch_size):
            batch_summaries = summaries[i:i + batch_size]
            batch_emb = self.embedding_model.embed_documents(batch_summaries)
            all_embeddings.extend(batch_emb)

        # 4. 转换为 numpy 数组并按列求均值，得到质心向量
        emb_matrix = np.array(all_embeddings)
        centroid = np.mean(emb_matrix, axis=0)

        print(f"[{self.lang}] 质心计算完成！")
        return centroid

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        dot_product = np.dot(vec1, vec2)
        norm_a = np.linalg.norm(vec1)
        norm_b = np.linalg.norm(vec2)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))

    def get_observation(self, current_summary: str, trajectory: Trajectory) -> tuple[str, float]:
        """
        根据当前生成的 summary 计算 observation。
        返回 (observation文本描述, 当前的相似度得分)
        """
        # 计算当前摘要的 embedding
        current_emb = np.array(self.embedding_model.embed_query(current_summary))

        # 计算与简洁簇质心的余弦相似度
        current_sim = self._cosine_similarity(current_emb, self.centroid)

        # 结合 Trajectory 分析趋势 (Trend)
        latest_record = trajectory.get_latest_record()

        if latest_record is None:
            trend_desc = "This is the initial draft. No previous trend available."
        else:
            prev_sim = latest_record.similarity_score
            diff = current_sim - prev_sim
            if diff > 0.01:
                trend_desc = f"Compared to the previous iteration (Score: {prev_sim:.4f}), your revision has moved CLOSER to the concise distribution. Keep refining in this direction."
            elif diff < -0.01:
                trend_desc = f"Compared to the previous iteration (Score: {prev_sim:.4f}), your revision has drifted AWAY from the concise distribution. Please reconsider your changes."
            else:
                trend_desc = f"Compared to the previous iteration (Score: {prev_sim:.4f}), the semantic distance remains mostly UNCHANGED."

        observation_text = (
            f"Observation from Reference Distribution:\n"
            f"The current summary's semantic proximity to the ideal concise cluster is {current_sim:.4f}.\n"
            f"Trend Analysis: {trend_desc}"
        )

        return observation_text, current_sim