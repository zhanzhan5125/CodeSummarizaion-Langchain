import copy
import json
import logging
import os
import warnings
from langchain.agents.agent import BaseSingleActionAgent
from model import CommentGenerator
from tool import CodeTools
import pandas as pd


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
        method = kwargs["method"]
        prompt_type = kwargs["prompt_type"]
        model_v = kwargs["model_v"]
        rag = kwargs["rag"]
        tools = kwargs["tools"]
        generator = kwargs["generator"]
        max_iter = kwargs["max_iter"]
        s_max = kwargs["s_max"]
        s_min = kwargs["s_min"]
        logger = kwargs["logger"]
        examples = []
        # data['file'] = tools[2].run({"lang": lang, "repo": data['repo'], "path": data['path'], 'sha': data['sha']})
        if rag != "no":
            print("[Retrieving examples...]")
            examples = tools[0].run({"data": data, "method": rag})
            # for example in examples:
            #     example["file"] = tools[2].run(
            #         {"lang": lang, "repo": example['data']['repo'], "path": example['data']['path'],
            #          "sha": example['data']['sha']})
            print("[Succeed Retrieved]")
        result = json.loads(generator.generate(data, examples))
        comment = result['Comment']
        original = comment
        logger.info(f"*** Original Comment ***: {original}")
        if method != "codeSummary":
            print("[Succeed generation]")
            return {
                "Comment": result['Comment'],
                'Origin': original}

        print('[Validating...]')
        score_data = tools[1].run({"code": code,
                                   "comment": comment,
                                   "model_v": model_v,
                                   "prompt_file": "./vaild_prompt/Conciseness.txt",
                                   "s_max": s_max,
                                   "s_min": s_min})
        # score_data = json.loads(generator.validate(code, comment))
        # result['Score'] = score_data['Conciseness']
        result['Conciseness'] = score_data['Conciseness']
        # result['Weighted'] = score_data['Weighted']
        result['Suggestion'] = score_data['Suggestion']

        logger.info(f"*** Conciseness Score ***: {result['Conciseness']}")
        # logger.info(f"*** Weighted Score ***: {result['Weighted']}")
        logger.info(f"*** Validation Suggestion ***: {result['Suggestion']}")

        for _ in range(max_iter):
            # if int(result['Score']) >= 7: break
            # if result['Weighted'] >= 8: break
            # first_two = result['Weighted'][:2]
            # if any(x >= 9 for x in first_two):
            #     break
            # if int(result['Conciseness']) == 10 :break
            if result['Suggestion'] == 'No suggestion matched.': break
            print('[Refining...]')
            logger.info(f"----------------Refine {_ + 1}----------------")

            rf_result = json.loads(generator.refine(code, comment, result['Suggestion']))
            # rf_result = json.loads(generator.score_refine(code, comment, str(result['Conciseness'])))
            comment = rf_result['Comment']
            logger.info(f"*** Refine comment ***: {comment}")

            score_data = tools[1].run({"code": code,
                                       "comment": comment,
                                       "model_v": model_v,
                                       "prompt_file": "./vaild_prompt/Conciseness.txt",
                                       "s_max": s_max,
                                       "s_min": s_min})

            logger.info(f"*** Conciseness Score ***: {score_data['Conciseness']}")
            # logger.info(f"*** Weighted Score ***: {score_data['Weighted']}")
            logger.info(f"*** Validation Suggestion ***: {score_data['Suggestion']}")

            rf_result['Conciseness'] = score_data['Conciseness']
            # rf_result['Weighted'] = score_data['Weighted']
            rf_result['Suggestion'] = score_data['Suggestion']

            result = rf_result

        logger.info(f"*** Final Score ***: {score_data['Conciseness']}")
        logger.info(f"*** Final Comment ***: {result['Comment']}")

        print("[Succeed generation]")
        return {
                "Score": result['Conciseness'],
                "Suggestion": result['Suggestion'],
                "Comment": result['Comment'],
                'Origin': original}


class AgentRunner:
    def __init__(self, llm_set, logger):
        self.llm_set = llm_set
        self.method = llm_set["method"]
        self.prompt_type = llm_set["prompt_type"]
        self.model_v = llm_set["model_v"]
        self.rag = llm_set["rag"]
        self.lang = llm_set["language"]
        self.max_iter = llm_set["max_iter"]
        self.s_max = llm_set["max_score"]
        self.s_min = llm_set["min_score"]
        self.logger = logger
        self.agent = CommentAgent()
        print("[Initializing AgentRunner]")

    def run(self, data: dict):
        tools = CodeTools(3, self.lang).tools
        generator = CommentGenerator(self.llm_set)
        result = self.agent.plan(tools=tools, data=data, language=self.lang, method=self.method,
                                 prompt_type=self.prompt_type, rag=self.rag, model_v=self.model_v,
                                 generator=generator, max_iter=self.max_iter, s_max=self.s_max, s_min=self.s_min,
                                 logger=self.logger)
        return result
