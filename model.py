import argparse
import logging
import os
from time import sleep
import warnings
import openai
from langchain_community.chat_models import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# os.environ["OPENAI_API_KEY"] = "your-api-key"  # 请替换为你的实际API密钥
# os.environ["OPENAI_BASE_URL"] = "your-base-url"  # 请替换为你的实际API密钥
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

BASIC_PROMPT = "Please generate a short comment(content only,ignore the format) in one sentence for the following function:\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

FEEDBACK_PROMPT = "Modify the comment in one sentence(content only,ignore the format) for the code based on score(scale of 0-5) and basis.\n"


# ANTHROPIC_API_KEY = "your-api-key"
# ANTHROPIC_BASE_URL = ""


class GPT:
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.basic_prompt = BASIC_PROMPT
        self.api_key = ''

    def ask(self, code, prompting, samples=''):
        print("llm asking......")
        if self.model_name == 'gpt-3.5-turbo':
            self.api_key = "sk-KPyLwEJBvLc6MYkgsiYIXT5l1L8GKOJn7aXrwmagns4mnaV9"
        elif self.model_name == 'gpt-4-1106-preview':
            self.api_key = "sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux"
        gpt = ChatOpenAI(model=self.model_name,
                         api_key=self.api_key,
                         base_url="https://xiaoai.plus/v1",
                         temperature=self.temperature,
                         model_kwargs={"top_p": self.top_p}
                         )
        memory = ConversationBufferMemory()
        memory.save_context({"HumanMessage": self.system_prompt}, {"AIMessage": ""})  # 无回应，系统提示为背景
        if prompting == 1:
            fewshots_example = samples
            for shot in fewshots_example:
                memory.save_context({"HumanMessage": shot['code']}, {"AIMessage": shot['comment']})
        conversation = ConversationChain(
            llm=gpt,
            memory=memory,
            verbose=True,
        )
        result = conversation.predict(input=self.basic_prompt + code)
        print("result: \n", result.strip())
        sleep(1)
        return result.strip()

    def feedback(self, code, answer, score, basis):
        score = str(score)
        if self.model_name == 'gpt-3.5-turbo':
            self.api_key = "sk-KPyLwEJBvLc6MYkgsiYIXT5l1L8GKOJn7aXrwmagns4mnaV9"
        elif self.model_name == 'gpt-4-1106-preview':
            self.api_key = "sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux"
        gpt = ChatOpenAI(model=self.model_name,
                         api_key=self.api_key,
                         base_url="https://xiaoai.plus/v1",
                         temperature=self.temperature,
                         model_kwargs={"top_p": self.top_p}
                         )
        messages = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
                    HumanMessage(
                        content=FEEDBACK_PROMPT + "\nCode:" + code + "\nComment:" + answer + "\nScore:" + score + "\nBasis:" + basis)]
        result = gpt(messages)
        result = result.content.strip()
        logger.info('FEEDBACK result:' + result)
        sleep(1)
        return result


class CLAUDE:
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.logger = args.logger
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.basic_prompt = BASIC_PROMPT

    def ask(self, code, prompting):
        claude = ChatAnthropic(model=self.model_name, temperature=self.temperature, top_p=self.top_p)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--openai_key", required=True, type=str)
    parser.add_argument("--model", default="gpt-3.5", type=str)
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    args = parser.parse_args()

    MODEL_NAME_OR_PATH = {'gpt-4': 'gpt-4-1106-preview',
                          'gpt-3.5': 'gpt-3.5-turbo',
                          # 'claude-3.5': 'claude-3.5'
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]

    model = GPT(args=args)
    code = "public Table newRow()\n    {\n        unnest();\n        nest(row = new Block(\"tr\"));\n        if (_defaultRow!=null)\n        {\n            row.setAttributesFrom(_defaultRow);\n            if (_defaultRow.size()>0)\n                row.add(_defaultRow.contents());\n        }\n        cell=null;\n        return this;\n    }"
    question = "def initialize_bars(self, sender=None, **kwargs):\n        \"\"\"Calls the initializers of all bound navigation bars.\"\"\"\n        for bar in self.bars.values():\n            for initializer in bar.initializers:\n                initializer(self)"

    model.ask(code, 0)

    model.ask(question, 0)
    # model.ask(code, 1)


if __name__ == '__main__':
    main()
