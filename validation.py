import csv
import json
import logging
import os
import re
import time
import warnings
import math
from langchain import requests
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from util.remove_comments import remove_comments_and_docstrings


def llm_score(code, comment, model, prompt_file, s_min=1, s_max=10, weight=False, temperature=0, top_p=1.0):
    prompt = open(prompt_file).read()
    cur_prompt = (prompt.replace('{Code}', code).replace('{Comment}', comment)
                  .replace('{S_min}', str(s_min)).replace('{S_max}', str(s_max)))
    if model == 'claude-3.7':
        llm = ChatAnthropic(
            model="claude-3-7-sonnet-20250219",
            api_key='sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7',
            # api_key = "sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V",
            # base_url="https://api.agicto.cn/v1",
            base_url="https://api.mjdjourney.cn",
            temperature=temperature,
            top_p=top_p,
        )
    else:
        if model == 'o3-mini':
            model = 'o3-mini-2025-01-31'
        llm = ChatOpenAI(model=model,
                         timeout=30,
                         api_key='sk-r5IQlRhuLFE5n5J7UVjNdqsYuFd83LHZQAPAMPXrANJApt6v',
                         base_url="https://xiaoai.plus/v1",
                         # api_key='sk-SpJc0amV97jp7BCK6aDd27Dd39D249529b28Ec8b70Ab9294',
                         # base_url="https://api.mjdjourney.cn/v1",
                         temperature=temperature,
                         top_p=top_p,
                         logprobs=True,
                         top_logprobs=5,
                         response_format={"type": "json_object"},
                         )
    messages = [HumanMessage(content=cur_prompt)]
    result = llm.generate([messages])
    # print(result)
    r = result.generations[0][0].text
    print(r)
    if model == 'claude-3.7':
        json_score = extract_single_json(r)
    else:
        json_score = json.loads(r)
    if not weight:
        print(json_score)
        # 把 key "a" 改成 "b"
        if "Concissness" in json_score:
            json_score["Conciseness"] = json_score.pop("Concissness")
        return json_score

    generated_info = result.generations[0][0].generation_info

    content = generated_info.get('logprobs').get('content')
    x = 100
    str_array = [str(i) for i in range(1, x + 1)]
    weight = []
    for c in content:
        token = str(c.get('token'))
        if token in map(str, json_score.values()):
            top_logprobs = c.get('top_logprobs')
            final_token = 0
            pro_scores = []
            for pro in top_logprobs:
                protoken = pro.get('token')
                logprob = pro.get('logprob')
                if protoken in str_array:
                    pro_scores.append(protoken)
                    final_token += int(protoken) * math.exp(logprob)
            print(pro_scores)
            weight.append(final_token)
    for key, new_val in zip(json_score.keys(), weight):
        json_score[key] = round(new_val, 4)
    print(json_score)
    # all_score = sum(float(v) for v in json_score.values()) / len(json_score.values())
    # json_score["OverAll"] = all_score
    # json_score['weight'] = round(weight[0], 4)
    # print(json_score)
    return json_score


def extract_single_json(text: str):
    """
    从给定的文本中提取唯一的 JSON 对象，并返回 Python 字典
    """
    # 规则1: 匹配包含特定字段的 JSON
    special_pattern = r"\{[^{}]*?(\"Conciseness\"[^{}]*?\"Suggestion\"|\"Suggestion\"[^{}]*?\"Conciseness\")[^{}]*?\}"
    match = re.search(special_pattern, text, re.DOTALL)
    if match:
        return json.loads(match.group(0))

    # 规则2: 普通 JSON 匹配
    general_pattern = r"\{[\s\S]*?\}"
    match = re.search(general_pattern, text)
    if not match:
        raise ValueError("未找到 JSON 对象")

    return json.loads(match.group(0))


if __name__ == '__main__':
    # testfile = "./Dataset/java/train/2000_java_train.jsonl"
    # f_v = open("./Result/trainset_validation/2000_java_valid.csv", "a", encoding="utf-8", newline='')
    # writer_v = csv.writer(f_v)
    # with open(testfile, "r", encoding="utf-8") as f:
    #     print("opening file ", testfile)
    #     cnt = 0
    #     for line in f:
    #         cnt += 1
    #         if cnt < 0 : continue
    #         line = line.strip()
    #         js = json.loads(line)
    #         try:
    #             code = remove_comments_and_docstrings(source=js['code'], lang='java')
    #         except Exception as e:
    #             print(e)
    #             writer_v.writerow([0,0,0])
    #             continue
    #         comment = ' '.join(js['docstring_tokens'])
    #         s = llm_score(code, comment, 'gpt-4-turbo', "./Evaluation/g-eval_prompt", True)
    #         print(s)
    #         writer_v.writerow([s['Comprehensiveness'], s['weight'], s["Thought process"]])
    #         # break
    # f_v.close()
    # code = '''
    # def headers_present(self, headers):
    #     headers = {name: re.compile('(.*)') for name in headers}
    #     self.add_matcher(matcher('HeadersMatcher', headers))
    # '''
    # comment_1 = "Add a matcher to the mock to check if specified headers are present in the request."
    # comment_2 = "Check presence of specified headers in request"
    # comment_3 = "Defines a list of headers that must be present in the outgoing request in order to satisfy the matcher no matter what value the headers hosts ."
    # llm_score(code, comment_1, "./Evaluation/e", True)
    # llm_score(code, comment_2, "./Evaluation/e", True)
    # llm_score(code, comment_3, "./Evaluation/e", True)
    #
    code = '''
def plot_rb_data(xdata, ydatas, yavg, yerr, fit, survival_prob, ax=None,
                 show_plt=True):
    if not HAS_MATPLOTLIB:
        raise ImportError('The function plot_rb_data needs matplotlib. '
                          'Run "pip install matplotlib" before.')
    if ax is None:
        plt.figure()
        ax = plt.gca()
    for ydata in ydatas:
        ax.plot(xdata, ydata, color='gray', linestyle='none', marker='x')
    ax.errorbar(xdata, yavg, yerr=yerr, color='r', linestyle='--', linewidth=3)
    ax.plot(xdata, survival_prob(xdata, *fit), color='blue', linestyle='-', linewidth=2)
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Clifford Length', fontsize=16)
    ax.set_ylabel('Z', fontsize=16)
    ax.grid(True)
    if show_plt:
        plt.show()
        '''
    comment_1 = "Trace the dependencies for app ."
    comment_2 = "Plot randomized benchmarking data ."
    start_time = time.time()
    print(llm_score(code, comment_2,"o3-mini-2025-01-31" ,"./vaild_prompt/Conciseness.txt", 1,10,False))
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算用时
    print(elapsed_time)
