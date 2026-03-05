from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TrajectoryRecord:
    """定义每一轮迭代的数据结构"""
    iteration: int
    code: str
    summary: str
    score: float
    observation: str
    reflection: str
    similarity_score: float


class Experience:
    """
    长期记忆 (Long-term Memory)：存放来自Empirical Study的经验
    """

    def __init__(self):
        # 提取自论文图5中的三条经验原则
        self.rules = [
            "1. A concise summary should begin with an action verb.",
            "2. A concise summary should convey functionality with as few tokens as possible with high information density.",
            "3. A concise summary should not describe specific implementation details."
        ]

    def get_experience_prompt(self) -> str:
        """将经验格式化为Prompt文本，供Validator打分时使用"""
        header = "Here are the experiences distilled from empirical studies for concise code summaries:\n"
        return header + "\n".join(self.rules)


class Trajectory:
    """
    短期记忆 (Short-term Memory)：存放多轮Refinement的历史轨迹
    """

    def __init__(self):
        self.records: List[TrajectoryRecord] = []

    def add_record(self, iteration: int, code: str, summary: str, score: float,
                   observation: str, reflection: str, similarity_score: float = 0.0):
        """添加一轮完整的迭代记录"""
        record = TrajectoryRecord(
            iteration=iteration,
            code=code,
            summary=summary,
            score=score,
            observation=observation,
            reflection=reflection,
            similarity_score=similarity_score
        )
        self.records.append(record)

    def get_latest_record(self) -> Optional[TrajectoryRecord]:
        """获取最新一轮的记录（用于提取上一步的相似度来计算Trend）"""
        if not self.records:
            return None
        return self.records[-1]

    def get_trajectory_history_prompt(self) -> str:
        """将历史轨迹转化为文本，供Generator或Reflector在下一轮参考"""
        if not self.records:
            return "No previous refinement history."

        history_str = "Previous Refinement Trajectory:\n"
        for rec in self.records:
            history_str += (
                f"--- Iteration {rec.iteration} ---\n"
                f"Summary Draft: {rec.summary}\n"
                f"Conciseness Score: {rec.score}\n"
                f"Reflection: {rec.reflection}\n"
            )
        return history_str