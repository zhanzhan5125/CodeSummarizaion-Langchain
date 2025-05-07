import copy
import json
import logging
import os
import warnings
from langchain.agents.agent import BaseSingleActionAgent
from model import CommentGenerator
from tool import CodeTools
import pandas as pd

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
log_file_path = os.path.join(os.path.join('./Result/generation/python/codeSummary', 'valid_ai4.txt'))
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
        result = json.loads(generator.generate(data, examples))
        comment = result['Comment']
        original = comment
        logger.info(f"*** Original Comment ***: {original}")

        print('[Validating...]')
        score_data = tools[1].run({"code": code,
                                   "comment": comment,
                                   "prompt_file": "./vaild_prompt/Conciseness.txt"})
        result['Score'] = score_data['Conciseness']
        result['Suggestion'] = score_data['Suggestion']

        logger.info(f"*** Validation Score ***: {result['Score']}")
        logger.info(f"*** Validation Suggestion ***: {result['Suggestion']}")

        for _ in range(max_iter):
            if int(result['Score']) == 5: break
            print('[Refining...]')
            logger.info(f"----------------Refine {_ + 1}----------------")

            rf_result = json.loads(generator.refine(code, comment, result['Suggestion']))
            comment = rf_result['Comment']
            logger.info(f"*** Refine comment ***: {comment}")

            score_data = tools[1].run({"code": code,
                                       "comment": comment,
                                       "prompt_file": "./vaild_prompt/Conciseness.txt"})

            logger.info(f"*** Validation Score ***: {score_data['Conciseness']}")
            logger.info(f"*** Validation Suggestion ***: {score_data['Suggestion']}")

            rf_result['Score'] = score_data['Conciseness']
            rf_result['Suggestion'] = score_data['Suggestion']

            result = rf_result
            # if int(result['Score']) >= 4: break

        logger.info(f"*** Final Score ***: {score_data['Conciseness']}")
        logger.info(f"*** Final Comment ***: {result['Comment']}")

        print("[Succeed generation]")
        return {"Score": result['Score'],
                "Suggestion": result['Suggestion'],
                "Comment": result['Comment'],
                'Origin': original}


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
