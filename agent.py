import copy
import json
import logging
import os
import warnings
from langchain.agents import BaseSingleActionAgent
from model import CommentGenerator
from tool import CodeTools
from Result.info import calculate_average, low_score

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
log_file_path = os.path.join(os.path.join('./Result/generation/python/codeSummary', 'test.txt'))
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)


class CommentAgent(BaseSingleActionAgent):
    def input_keys(self):
        pass

    def aplan(self):
        pass

    def plan(self, **kwargs):
        # 第一次生成
        data = kwargs["data"]
        code = data["code"]
        lang = kwargs["language"]
        prompt_type = kwargs["prompt_type"]
        rag = kwargs["rag"]
        tools = kwargs["tools"]
        generator = kwargs["generator"]
        max_iter = kwargs["max_iter"]
        examples = []
        if rag != "no":
            print("[Retrieving examples...]")
            examples = tools[0].run({"data": data, "method": rag})
            print("[Succeed Retrieved]")
        result = generator.generate(data, examples)
        if prompt_type == "ASAP" or prompt_type == "SCSL":
            result = {'Comment': result, 'Thought process': ""}
        # print(result)
        comment = result['Comment']
        logger.info(f"*** Original Comment ***: {comment}")
        # logger.info(f"*** Original Thought ***: {result['Thought process']}")
        # 评分循环Thought process
        file = "./Result/trainset_validation/10000_python_valid.csv"
        standard = calculate_average(file, 1, low_score(file, 1))
        print('[Validating...]')
        score_data = tools[1].run({"code": code,
                                   "comment": comment,
                                   "prompt_file": "./vaild_prompt/Comprehensiveness_cot.txt"})
        result['Score'] = score_data['Weighted']
        result['Validation thought'] = score_data['Thought process']

        logger.info(f"*** Validation Score ***: {result['Score']}")
        logger.info(f"*** Validation Thought ***: {result['Validation thought']}")

        for _ in range(max_iter):
            if prompt_type == "codeSummary" and result['Score'] <= 4.3:
                print("[Regenerating comment...]")
                logger.info(f"----------------Iteration {_ + 1}----------------")
                fd_result = json.loads(
                    generator.feedback(code, comment, str(result['Score']), result['Validation thought']))
                comment = fd_result['Comment']
                logger.info(f"*** Feedback Comment ***: {comment}")
                # logger.info(f"*** Feedback Thought ***: {fd_result['Thought process']}")
                print('[Validating...]')
                score_data = tools[1].run({"code": code,
                                           "comment": comment,
                                           "prompt_file": "./vaild_prompt/Comprehensiveness_cot.txt"})

                logger.info(f"*** Validation Score ***: {score_data['Weighted']}")
                logger.info(f"*** Validation Thought ***: {score_data['Thought process']}")

                fd_result['Score'] = score_data['Weighted']
                fd_result['Validation thought'] = score_data['Thought process']

                if fd_result['Score'] >= result['Score']:
                    result = copy.deepcopy(fd_result)
            else:
                break

        print("[Succeed generation]")

        return result


class AgentRunner:
    def __init__(self, llm_set):
        self.llm_set = llm_set
        self.prompt_type = llm_set["prompt_type"]
        self.rag = llm_set["rag"]
        self.lang = llm_set["language"]
        self.max_iter = llm_set["max_iter"]
        self.agent = CommentAgent()
        print("[Initializing AgentRunner]")

    def run(self, data: dict):
        tools = CodeTools(3, self.lang).tools
        generator = CommentGenerator(self.llm_set)
        result = self.agent.plan(tools=tools, data=data, language=self.lang, prompt_type=self.prompt_type, rag=self.rag,
                                 generator=generator, max_iter=self.max_iter)
        return result
