import sys
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
import logging
import warnings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import anthropic

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

BASIC_PROMPT = "Please generate a short comment(content only,ignore the format and punctuations) in one sentence for the following function:\n"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
FEEDBACK_PROMPT = "Modify the comment in one sentence(content only,ignore the format and punctuations) for the code based on score(scale of 0-5) and basis.\n"


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
            self.api_key = "sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux"
        elif self.model == "gpt-3.5":
            self.model_name = 'gpt-3.5-turbo'
            self.api_key = "sk-KPyLwEJBvLc6MYkgsiYIXT5l1L8GKOJn7aXrwmagns4mnaV9"

        llm = ChatOpenAI(model=self.model_name,
                         api_key=self.api_key,
                         base_url="https://xiaoai.plus/v1",
                         temperature=self.temperature,
                         top_p=self.top_p
                         )
        return llm

    def claude(self):
        self.model_name = "claude-3-5-sonnet-20240620"
        self.api_key = "sk-eTJRFYo2sIgZE1Klmwanuq0gOQbjJ73TxEHExTuIRChMmliZ"
        llm = ChatAnthropic(
            model="claude-3-5-sonnet-20240620",
            api_key=self.api_key,
            base_url="https://anthropic.claude-plus.top",
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
            logger.info(f"ERROR!No model named {self.model}")
            sys.exit(1)


class CommentGenerator:

    def __init__(self, llm_set):
        self.chain = None
        logger.info("Initializing CommentGenerator")
        self.prompt_type = llm_set['prompt_type']
        self.memory = ConversationBufferMemory()
        self.llm_set = llm_set

        self.llm = LLM(self.llm_set).generate_llm()

        # 无回应，系统提示为背景
        self.memory.save_context({"HumanMessage": DEFAULT_SYSTEM_PROMPT}, {"AIMessage": ""})

        # 定义不同提示策略
        self.prompt_templates = {
            "zero_shot": ChatPromptTemplate.from_template(
                "{basic_prompt_code}"
            ),
            "few_shot": ChatPromptTemplate.from_messages(
                [
                    ("human", "{basic_prompt_code}")
                ])
        }

        logger.info("Succeeded CommentGenerator")

    def generate(self, code, examples=None):
        logger.info("Generating comment....")
        if self.prompt_type == "few_shot" and examples:
            for ex in examples:
                self.memory.save_context(
                    {"input_key": ex["code"]},
                    {"output_key": ex["comment"]}
                )

        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_templates[self.prompt_type],
            memory=self.memory
        )
        inputs = BASIC_PROMPT + "\n" + code
        result = self.chain({'basic_prompt_code': inputs})
        return result["text"]

    def feedback(self, code, answer, score, basis):
        logger.info("Feedbacking comment....")
        inputs = FEEDBACK_PROMPT + "\nScore:" + score + "\nBasis:" + basis
        result = self.chain({'basic_prompt_code': inputs})

        return result['text']


if __name__ == "__main__":
    # llm = ChatAnthropic(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=1024,
    #     anthropic_api_key="sk-UFaswAeaNrZTadFgf3rTOWg0veOmWZ5T180CPLTILwjvgnXV",
    #     base_url="https://xiaoai.plus",
    # )
    #
    # print(llm("hello"))

    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key="sk-eTJRFYo2sIgZE1Klmwanuq0gOQbjJ73TxEHExTuIRChMmliZ",
        base_url="https://anthropic.claude-plus.top",
    )
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        messages=[
            {"role": "user", "content": "Hello, Claude"}
        ],
        max_tokens=1024
    )
    print(message)
