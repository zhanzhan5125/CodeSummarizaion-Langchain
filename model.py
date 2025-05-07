import json
import sys
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import logging
import warnings
from langchain_core.messages import SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from asap_model import asap

ASAP_SYSTEM_PROMPT = '''
{Code}

Please find some info about the location of the function in the repo.
{Repo_info}

Please find the dataflow of the function.
We present the source and list of target indices.
{Dataflow}

We categorized the identifiers into different classes.
Please find the information below.
{Scopes}

Output comment
{Comment}
'''

SCSL_SYSTEM_PROMPT = '''
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design.
Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
Please ensure that your responses are socially unbiased and positive in nature.
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information.
'''

SCSL_BASIC_PROMPT = '''
Please generate a short comment in one sentence for the following function:
{Code}
'''

BASIC_PROMPT = '''
Make sure you still remember and understand instructions above carefully.
Return the comment meets the requirements in json format.

Code:
{Code}

Output Form: 
{"Comment":""}
'''

DEFAULT_SYSTEM_PROMPT = """\
You are an expert {Language} code assistant with a deep understanding of {Language} programming.
Your task is to analyze the provided {Language} code snippet steps by steps and create a comment.

Summarization Steps:
1. Read the source code and its component/structure carefully.
2. Identify the main function and key points.
3. Generate a most concentrate comment in a sentence.
"""
FEEDBACK_PROMPT = """\
According to criteria, here are the evaluation score and thought process for generating code comment.
Make sure you still remember and understand instructions above carefully.

Please refine the comment to approach 5 points.

Comprehensive Criteria (scale of 1 to 5):
A comprehensive comment should be:
1. Correct: Accurately summarizes what the code mainly does without logical error.
2. Concise: ONLY Focus on functional purpose in a high-level, abstract manner, avoid process details.
3. Clear: Enable the reader to quickly grasp the overall intent of the code.

Evaluation score: 
{Score}

Evaluation thought process: 
{ThoughtProcess}

Output JSON Form: 
{"Comment":""}

"""


class LLM:
    def __init__(self, llm_set=None):
        self.api_key = None
        self.model_name = None
        self.model = llm_set["model"]
        self.temperature = llm_set["temperature"]
        self.top_p = llm_set["top_p"]

    def gpt(self):
        if self.model == "gpt-4":
            self.model_name = 'gpt-4-1106-preview'
        elif self.model == "gpt-3.5":
            self.model_name = 'gpt-3.5-turbo'

        llm = ChatOpenAI(model=self.model_name,
                         api_key='sk-UFaswAeaNrZTadFgf3rTOWg0veOmWZ5T180CPLTILwjvgnXV',
                         base_url="https://xiaoai.plus/v1",
                         # api_key='sk-SpJc0amV97jp7BCK6aDd27Dd39D249529b28Ec8b70Ab9294',
                         # base_url="https://api.mjdjourney.cn/v1",
                         temperature=self.temperature,
                         top_p=self.top_p,
                         timeout=30,
                         response_format={"type": "json_object"}
                         )
        return llm

    def claude(self):
        self.model_name = "claude-3-5-sonnet-20240620"
        self.api_key = "sk-s7LCrJ8DONsBK9vv92Cf64Aa0cF74f6bAaF9727cBa01002f"
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            api_key=self.api_key,
            base_url="https://api.mjdjourney.cn",
            temperature=self.temperature,
            top_p=self.top_p
        )
        return llm

    def generate_llm(self):
        if self.model == "gpt-4" or self.model == "gpt-3.5":
            return self.gpt()
        elif self.model == "claude-3.5":
            return self.claude()
        else:
            print(f"ERROR!No model named {self.model}")
            sys.exit(1)


class CommentGenerator:
    def __init__(self, llm_set):
        self.chain = None
        print("[Initializing CommentGenerator...]")
        self.prompt_type = llm_set['prompt_type']
        self.llm_set = llm_set
        self.llm = LLM(self.llm_set).generate_llm()
        self.lang = llm_set['language']
        # 定义不同提示策略
        self.prompt_templates = {
            "SCSL": ChatPromptTemplate.from_messages([
                # 历史对话从 memory 自动注入
                ("system", SCSL_SYSTEM_PROMPT),  # 可选系统消息
                MessagesPlaceholder(variable_name="chat_history"),  # 关键：此处加载 memory
                ("human", "{basic_prompt_code}"),  # 当前用户输入
            ]),
            "codeSummary": ChatPromptTemplate.from_messages([
                # 历史对话从 memory 自动注入
                ("system", DEFAULT_SYSTEM_PROMPT.replace('{Language}', str(self.lang))),  # 可选系统消息
                MessagesPlaceholder(variable_name="chat_history"),  # 关键：此处加载 memory
                ("human", "{basic_prompt_code}"),  # 当前用户输入
            ]),
            "ASAP": ChatPromptTemplate.from_messages([
                # MessagesPlaceholder(variable_name="chat_history"),  # 关键：此处加载 memory
                ("human", "{basic_prompt_code}"),  # 当前用户输入
            ]),
        }
        # 初始化 memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 必须与模板中的 variable_name 匹配
            return_messages=True  # 若使用 ChatModel（如 GPT），需设为 True
        )
        self.asap_bm25 = ""
        print("[Succeeded CommentGenerator]")

    def generate(self, data, examples=None):
        code = data['code']
        print("[Generating comment....]")
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_templates[self.prompt_type],
            memory=self.memory,
            verbose=True
        )
        # print(self.memory.chat_memory.messages)
        result = {}
        if examples is not None:
            for ex in examples:
                infos = ex["code"]
                out = ex["data"]["comment"]
                if self.prompt_type == "ASAP":
                    info = asap(self.lang, ex["data"])
                    self.asap_bm25 = (ASAP_SYSTEM_PROMPT.replace('{Code}', info["Code"])
                                      .replace('{Repo_info}', info['Repo_info'])
                                      .replace('{Dataflow}', info['Dataflow'])
                                      .replace('{Scopes}', info['Scopes'])
                                      .replace('{Comment}', out)
                                      )
                self.memory.save_context(
                    {"input_key": infos},
                    {"output_key": out}
                )
        if self.prompt_type == "codeSummary":
            inputs = BASIC_PROMPT.replace('{Code}', code).replace('{Language}', str(self.lang))
            result = self.chain({'basic_prompt_code': inputs})
        elif self.prompt_type == "SCSL":
            inputs = SCSL_BASIC_PROMPT.replace('{Code}', code)
            result = self.chain({'basic_prompt_code': inputs})
        elif self.prompt_type == "ASAP":
            input = asap(self.lang, data)
            inputs = (ASAP_SYSTEM_PROMPT.replace('{Code}', code)
                      .replace('{Repo_info}', input['Repo_info'])
                      .replace('{Dataflow}', input['Dataflow'])
                      .replace('{Scopes}', input['Scopes'])
                      .replace('{Comment}', ""))
            result = self.chain({'basic_prompt_code': self.asap_bm25 + inputs})
        print(self.memory.chat_memory.messages[-1])
        return json.loads(result["text"])

    def feedback(self, code, comment, score, thought):
        print("Feedbacking comment....")
        self.memory.chat_memory.messages = []
        self.memory.save_context(
            {"input_key": code},
            {"output_key": comment}
        )
        # if len(messages) >= 2:
        #     del messages[-1]
        inputs = FEEDBACK_PROMPT.replace('{Code}', code).replace('{Comment}', comment).replace('{Score}',score).replace('{ThoughtProcess}', thought)
        result = self.chain({'basic_prompt_code': inputs})
        return result['text']
