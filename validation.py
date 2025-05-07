import csv
import json
import logging
import os
import time
import warnings
import math

from langchain import requests
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from util.remove_comments import remove_comments_and_docstrings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def llm_score(code, comment, prompt_file, temperature=0.1, top_p=1.0):
    prompt = open(prompt_file).read()
    cur_prompt = prompt.replace('{Code}', code).replace('{Comment}', comment)

    gpt = ChatOpenAI(model='gpt-4-1106-preview',
                     timeout=30,
                     api_key='sk-UFaswAeaNrZTadFgf3rTOWg0veOmWZ5T180CPLTILwjvgnXV',
                     base_url="https://xiaoai.plus/v1",
                     # api_key='sk-SpJc0amV97jp7BCK6aDd27Dd39D249529b28Ec8b70Ab9294',
                     # base_url="https://api.mjdjourney.cn/v1",
                     temperature=temperature,
                     top_p=top_p,
                     logprobs=True,
                     top_logprobs=5,
                     response_format={"type": "json_object"},
                     # model_kwargs={'top_p': top_p},
                     )

    messages = [SystemMessage(content=cur_prompt)]
    # print(messages)
    result = gpt.generate([messages])
    # print(result)
    generated_info = result.generations[0][0].generation_info

    content = generated_info.get('logprobs').get('content')
    top_logprobs = generated_info.get('logprobs').get('content')[0].get('top_logprobs')
    for c in content:
        if c.get('token') in {'1', '2', '3', '4', '5'}:
            top_logprobs = c.get('top_logprobs')
            break
    final_token = 0
    for pro in top_logprobs:
        protoken = pro.get('token')
        logprob = pro.get('logprob')
        if protoken in {'1', '2', '3', '4', '5'}:
            final_token += int(protoken) * math.exp(logprob)

    final = json.loads(result.generations[0][0].text)
    final["Weighted"] = final_token
    # logger.info(f"TOP_LOGPROBS {top_logprobs}")
    # logger.info(f"Validation Results: {final}")
    return final


if __name__ == '__main__':
    data_dir = "./Dataset/python/test"
    data_file = "187_python_test.jsonl"
    data_file_path = os.path.join(data_dir, data_file)

    out_dir = "./Result/testset_validation"
    out_file = "187_python_valid.csv"
    out_file_path = os.path.join(out_dir, out_file)
    o = open(out_file_path, 'a', encoding="utf-8", newline='')
    writer = csv.writer(o)
    idx = 0
    cnt = 91

    log_file_path = os.path.join(os.path.join(out_dir, 'log.txt'))
    fh = logging.FileHandler(log_file_path)

    logger.addHandler(fh)  # add the handlers to the logger
    logger.info(f'[Validation] ---- [{data_file}]')

    with open(data_file_path, "r", encoding="utf-8") as f:
        for line in f:
            idx += 1
            if idx < cnt: continue
            logger.info(f"------------------[{idx}]--------------------")
            js = json.loads(line.strip())
            code = remove_comments_and_docstrings(source=js['code'], lang='python')
            comment = ' '.join(js['docstring_tokens'])
            result = llm_score(code=code, comment=comment, prompt_file="./vaild_prompt/Comprehensiveness_cot.txt")
            writer.writerow([result["Comprehensiveness"], result["Weighted"], result["Thought process"]])

    o.close()

