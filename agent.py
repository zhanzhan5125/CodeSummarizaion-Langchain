import logging
import warnings
from langchain.agents import BaseSingleActionAgent
from model1 import CommentGenerator
from tool import CodeTools

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class CommentAgent(BaseSingleActionAgent):

    def input_keys(self):
        return ["code", "lang", "prompt_type"]

    def aplan(self):
        pass

    def plan(self, **kwargs):
        # 第一次生成
        code = kwargs["code"]
        lang = kwargs["language"]
        prompt_type = kwargs["prompt_type"]
        tools = kwargs["tools"]
        generator = kwargs["generator"]
        max_iter = kwargs["max_iter"]
        examples = []
        if prompt_type == "few_shot":
            logger.info("Retrieving examples")
            examples = tools[0].run({"code": code, "lang": lang})
            logger.info("Succeed Retrieved")

        comment = generator.generate(code, examples)
        logger.info(f"Original comment {comment}")
        # 评分循环
        for _ in range(max_iter):
            score_data = tools[1].run({"code": code, "comment": comment})
            logger.info(f"Score_data {score_data}")
            if int(score_data["score"]) == 5:
                return {"final_output": comment}
            # 反馈重新生成
            comment = generator.feedback(code, comment, score_data["score"], score_data["basis"])
            logger.info(f"Feedback comment {comment}")
        return {"final_output": comment}


class AgentRunner:
    def __init__(self, llm_set):
        logger.info("AgentRunner.......")
        self.llm_set = llm_set
        self.prompt_type = llm_set["prompt_type"]
        self.lang = llm_set["language"]
        self.max_iter = llm_set["max_iter"]
        self.generator = CommentGenerator(llm_set)
        self.agent = CommentAgent()

    def run(self, code: str):
        tools = CodeTools().tools
        result = self.agent.plan(tools=tools, code=code, language=self.lang, prompt_type=self.prompt_type,
                                 generator=self.generator, max_iter=self.max_iter)
        return result
