import json
import sys
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from Baseline.ASAP import ASAP_SYSTEM_PROMPT
from Baseline.SCSL import SCSL_generate
from Baseline.asap_model import asap
from prompts.llms import DEFAULT_SYSTEM_PROMPT, BASIC_PROMPT, REFINE_PROMPT


class LLM:
    def __init__(self, llm_set=None):
        self.api_key = None
        self.model_name = None
        self.model = llm_set["model"]
        self.temperature = llm_set["temperature"]
        self.top_p = llm_set["top_p"]

    def gpt(self):
        if self.model == "gpt-4-turbo":
            self.model_name = 'gpt-4-turbo'
        elif self.model == "gpt-4":
            self.model_name = 'gpt-4'
        elif self.model == "o3-mini":
            self.model_name = 'o3-mini-2025-01-31'
        elif self.model == "gpt-4o":
            self.model_name = 'gpt-4o'

        llm = ChatOpenAI(model=self.model_name,
                         api_key='sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
                         base_url="https://api.agicto.cn/v1",
                         # api_key='sk-SpJc0amV97jp7BCK6aDd27Dd39D249529b28Ec8b70Ab9294',
                         # base_url="https://api.mjdjourney.cn/v1",
                         # api_key='sk-UFaswAeaNrZTadFgf3rTOWg0veOmWZ5T180CPLTILwjvgnXV',
                         # base_url="https://xiaoai.plus/v1",
                         temperature=self.temperature,
                         top_p=self.top_p,
                         # timeout=30,
                         # response_format={"type": "json_object"}
                         )
        return llm

    def claude(self):
        self.model_name = "claude-3-7-sonnet-20250219"
        self.api_key = 'sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V'
        # self.api_key = 'sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7'
        llm = ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            base_url="https://api.agicto.cn/v1",
            # base_url="https://api.mjdjourney.cn",
            temperature=self.temperature,
            top_p=self.top_p,
        )
        return llm

    def generate_llm(self):
        if self.model == "gpt-4-turbo" or self.model == "o3-mini" or self.model == "gpt-4o":
            return self.gpt()
        elif self.model == "claude-3.7":
            return self.claude()
        else:
            print(f"ERROR!No model named {self.model}")
            sys.exit(1)


class CommentGenerator:
    def __init__(self, llm_set):
        self.chain = None
        print("[Initializing CommentGenerator...]")
        self.prompt_type = llm_set['prompt_type']
        self.method = llm_set['method']
        self.model = llm_set["model"]
        self.llm_set = llm_set
        self.llm = LLM(self.llm_set).generate_llm()
        self.lang = llm_set['language']
        # 定义不同提示策略
        self.prompt_templates = {
            "codeSummary": ChatPromptTemplate.from_messages([
                # 历史对话从 memory 自动注入
                ("system", DEFAULT_SYSTEM_PROMPT.replace('{Language}', str(self.lang))),  # 可选系统消息
                MessagesPlaceholder(variable_name="chat_history"),  # 关键：此处加载 memory
                ("human", "{basic_prompt_code}"),  # 当前用户输入
            ]),
            "ASAP": ChatPromptTemplate.from_messages([
                MessagesPlaceholder(variable_name="chat_history"),  # 关键：此处加载 memory
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
        if self.method == "SCSL":
            result = SCSL_generate(code, self.lang, self.prompt_type, self.model)
            return json.dumps({'Comment': result})
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_templates[self.method],
            memory=self.memory,
            verbose=True
        )
        result = {}
        ex_inputs = ''
        if examples is not None:
            # print(examples)
            ex_inputs += "Examples:\n"
            for ex in examples:
                ex_out = ex["data"]["comment"]
                context = {
                    'Repo': ex["data"]['repo'],
                    'Path': ex["data"]['path'],
                    'Function_name': ex["data"]['func'],
                    # 'Source file': ex['file']
                }
                context = '\n'.join([f'{k}: {v}' for k, v in context.items()])
                ex_inputs = BASIC_PROMPT.replace('{Code}', ex["data"]["code"]).replace('{Context}', context)
                # ex_inputs = BASIC_PROMPT_NO_CONTEXT.replace('{Code}', ex["data"]["code"])
                if self.method == "ASAP":
                    info = asap(self.lang, ex["data"])
                    ex_inputs = (ASAP_SYSTEM_PROMPT.replace('{Code}', info["Code"])
                                 .replace('{Repo_info}', info['Repo_info'])
                                 .replace('{Dataflow}', info['Dataflow'])
                                 .replace('{Scopes}', info['Scopes'])
                                 # .replace('{Comment}', ex_out)
                                 )
                self.memory.save_context(
                    {"input_key": ex_inputs},
                    {"output_key": ex_out}
                )

        if self.method == "codeSummary":
            context = {
                'Repo': data['repo'],
                'Path': data['path'],
                'Function_name': data['func'],
                # 'Source file': data['file']
            }
            context = '\n'.join([f'{k}: {v}' for k, v in context.items()])
            inputs = BASIC_PROMPT.replace('{Code}', code).replace('{Context}', context)
            # inputs = BASIC_PROMPT_NO_CONTEXT.replace('{Code}', code).replace('{Context}', context)

            result = self.chain({'basic_prompt_code': inputs})
            return json.dumps({'Comment': result["text"]})
        elif self.method == "ASAP":
            input = asap(self.lang, data)
            inputs = (ASAP_SYSTEM_PROMPT.replace('{Code}', code)
                      .replace('{Repo_info}', input['Repo_info'])
                      .replace('{Dataflow}', input['Dataflow'])
                      .replace('{Scopes}', input['Scopes'])
                      .replace('{Comment}', ""))
            result = self.chain({'basic_prompt_code': inputs})
            return json.dumps({'Comment': result["text"]})

    # def validate(self, code, comment):
    #     print("Refining comment....")
    #     prompt = open('prompts/validator.txt').read()
    #     cur_prompt = prompt.replace('{Code}', code).replace('{Comment}', comment)
    #     messages = [HumanMessage(content=cur_prompt)]
    #     result = self.llm.generate([messages])
    #     print(result)
    #     return json.dumps({'Suggestion': result.generations[0][0].text, 'Conciseness': 0})

    # def reflector(self, code, comment, observation, score):
    #     print("Reflecting comment....")


    def refine(self, code, comment, suggestion):
        print("Refining comment....")
        inputs = REFINE_PROMPT.replace('{Suggestion}', suggestion).replace('{Code}', code).replace('{Comment}', comment)
        # print(inputs)
        messages = [HumanMessage(content=inputs)]
        result = self.llm.generate([messages])
        # result = self.chain(inputs)
        # return json.dumps({'Comment': result['text']})
        return json.dumps({'Comment': result.generations[0][0].text})

    # def score_refine(self, code, comment, score):
    #     print("Refining comment....")
    #     inputs = Score_REFINE_PROMPT.replace('{Score}', score).replace('{Code}', code).replace('{Comment}', comment)
    #     # print(inputs)
    #     messages = [HumanMessage(content=inputs)]
    #     result = self.llm.generate([messages])
    #     # result = self.chain(inputs)
    #     # return json.dumps({'Comment': result['text']})
    #     return json.dumps({'Comment': result.generations[0][0].text})


