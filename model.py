import json
import sys
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage

from Baseline.SCSL import SCSL_generate
from Baseline.asap_model import asap

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
Code:
{Code}

Please find some info about the context given:
{Context}

Output Comment ONLY (without Comment symbols)
'''

BASIC_PROMPT_NO_CONTEXT = '''
Code:
{Code}

Output Comment ONLY (without Comment symbols)
'''

DEFAULT_SYSTEM_PROMPT = """\
You are an expert {Language} code assistant with a deep understanding of {Language} programming.
Your task is to analysis given function and create a complete docstring for it.
"""


REFINE_PROMPT = '''\
Please fix the comment based on the suggestions given.

Code:
{Code}

Original Comment:
{Comment}

Suggestion:
{Suggestion}

Output Comment ONLY (without Comment symbols)
'''

Score_REFINE_PROMPT = '''\
Please fix the comment based on the Conciseness Score given.

Code:
{Code}

Original Comment:
{Comment}

Score:
{Score}

Output Comment ONLY (without Comment symbols)
'''

fewshot_example_language = {
    'java': [
        {'code':"@Override\n    public ImageSource apply(ImageSource input) {\n        final int[][] pixelMatrix = new int[3][3];\n\n        int w = input.getWidth();\n        int h = input.getHeight();\n\n        int[][] output = new int[h][w];\n\n        for (int j = 1; j < h - 1; j++) {\n            for (int i = 1; i < w - 1; i++) {\n                pixelMatrix[0][0] = input.getR(i - 1, j - 1);\n                pixelMatrix[0][1] = input.getRGB(i - 1, j);\n                pixelMatrix[0][2] = input.getRGB(i - 1, j + 1);\n                pixelMatrix[1][0] = input.getRGB(i, j - 1);\n                pixelMatrix[1][2] = input.getRGB(i, j + 1);\n                pixelMatrix[2][0] = input.getRGB(i + 1, j - 1);\n                pixelMatrix[2][1] = input.getRGB(i + 1, j);\n                pixelMatrix[2][2] = input.getRGB(i + 1, j + 1);\n\n                int edge = (int) convolution(pixelMatrix);\n                int rgb = (edge << 16 | edge << 8 | edge);\n                output[j][i] = rgb;\n            }\n        }\n\n        MatrixSource source = new MatrixSource(output);\n        return source;\n    }", 'nl': "Expects a height mat as input"}
        ,{'code':"public static ComplexNumber Add(ComplexNumber z1, ComplexNumber z2) {\r\n        return new ComplexNumber(z1.real + z2.real, z1.imaginary + z2.imaginary);\r\n    }", 'nl': "Adds two complex numbers."}
        ,{'code':"public void setOutRGB(IntRange outRGB) {\r\n        this.outRed = outRGB;\r\n        this.outGreen = outRGB;\r\n        this.outBlue = outRGB;\r\n\r\n        CalculateMap(inRed, outRGB, mapRed);\r\n        CalculateMap(inGreen, outRGB, mapGreen);\r\n        CalculateMap(inBlue, outRGB, mapBlue);\r\n    }", 'nl': "Set RGB output range."}
        ,{'code':"public Table newRow()\n    {\n        unnest();\n        nest(row = new Block(\"tr\"));\n        if (_defaultRow!=null)\n        {\n            row.setAttributesFrom(_defaultRow);\n            if (_defaultRow.size()>0)\n                row.add(_defaultRow.contents());\n        }\n        cell=null;\n        return this;\n    }", 'nl': 'Create new table row . Attributes set after this call and before a call to newCell or newHeader are considered row attributes.'}
    ],
    'python': [
        {'code': 'def get_meshes_fld(step, var):\n    fld = step.fields[var]\n    if step.geom.twod_xz:\n        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]\n        fld = fld[:, 0, :, 0]\n    elif step.geom.cartesian and step.geom.twod_yz:\n        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    else:  \n        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    return xmesh, ymesh, fld', 'nl': 'Return scalar field along with coordinates meshes .'}
        ,{'code': 'def _obtain_token(self):\n        if self.expiration and self.expiration > datetime.datetime.now():\n            return\n        resp = requests.post("{}/1.1/oauth/token".format(API_URL), data={\n            "client_id": self.client_id,\n            "client_secret": self.client_secret,\n            "grant_type": "client_credentials"\n        }).json()\n        if "error" in resp:\n            raise APIError("LibCal Auth Failed: {}, {}".format(resp["error"], resp.get("error_description")))\n        self.expiration = datetime.datetime.now() + datetime.timedelta(seconds=resp["expires_in"])\n        self.token = resp["access_token"]\n        print(self.token)', 'nl': 'Obtain an auth token from client id and client secret .'}
        ,{'code': "def get_byte(self, i):\n        value = []\n        for x in range(2):\n            c = next(i)\n            if c.lower() in _HEX:\n                value.append(c)\n            else:  \n                raise SyntaxError('Invalid byte character at %d!' % (i.index - 1))\n        return ''.join(value)", 'nl': 'Get byte .'}
        ,{'code': "def get_item_children(item):\r\n    children = [item.child(index) for index in range(item.childCount())]\r\n    for child in children[:]:\r\n        others = get_item_children(child)\r\n        if others is not None:\r\n            children += others\r\n    return sorted(children, key=lambda child: child.line)", 'nl': "Return a sorted list of all the children items of item ."}
    ]}
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
                         # api_key='sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
                         # base_url="https://api.agicto.cn/v1",
                         api_key='sk-SpJc0amV97jp7BCK6aDd27Dd39D249529b28Ec8b70Ab9294',
                         base_url="https://api.mjdjourney.cn/v1",
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
        # self.api_key = 'sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V'
        self.api_key = 'sk-ASEYrFoHwCdfmCJa67CfA6C2E9F0446b97Bd1103Fd7c1aE7'
        llm = ChatAnthropic(
            model=self.model_name,
            api_key=self.api_key,
            # base_url="https://api.agicto.cn/v1",
            base_url="https://api.mjdjourney.cn",
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
            result = (code, self.lang, self.prompt_type, self.model)
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

    def validate(self, code, comment):
        print("Refining comment....")
        prompt = open('./vaild_prompt/Conciseness.txt').read()
        cur_prompt = prompt.replace('{Code}', code).replace('{Comment}', comment)
        messages = [HumanMessage(content=cur_prompt)]
        result = self.llm.generate([messages])
        print(result)
        return json.dumps({'Suggestion': result.generations[0][0].text, 'Conciseness': 0})

    def refine(self, code, comment, suggestion):
        print("Refining comment....")
        inputs = REFINE_PROMPT.replace('{Suggestion}', suggestion).replace('{Code}', code).replace('{Comment}', comment)
        # print(inputs)
        messages = [HumanMessage(content=inputs)]
        result = self.llm.generate([messages])
        # result = self.chain(inputs)
        # return json.dumps({'Comment': result['text']})
        return json.dumps({'Comment': result.generations[0][0].text})

    def score_refine(self, code, comment, score):
        print("Refining comment....")
        inputs = Score_REFINE_PROMPT.replace('{Score}', score).replace('{Code}', code).replace('{Comment}', comment)
        # print(inputs)
        messages = [HumanMessage(content=inputs)]
        result = self.llm.generate([messages])
        # result = self.chain(inputs)
        # return json.dumps({'Comment': result['text']})
        return json.dumps({'Comment': result.generations[0][0].text})


