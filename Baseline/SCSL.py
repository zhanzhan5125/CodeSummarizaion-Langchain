from time import sleep
import anthropic
import openai
from anthropic import HUMAN_PROMPT, AI_PROMPT, Anthropic
from openai import OpenAI

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""

BASIC_PROMPT = "Please generate a short comment in one sentence for the following function:\n"

fewshot_example_language_4 = {
    'java': [
        {
            'code': "@Override\n    public ImageSource apply(ImageSource input) {\n        final int[][] pixelMatrix = new int[3][3];\n\n        int w = input.getWidth();\n        int h = input.getHeight();\n\n        int[][] output = new int[h][w];\n\n        for (int j = 1; j < h - 1; j++) {\n            for (int i = 1; i < w - 1; i++) {\n                pixelMatrix[0][0] = input.getR(i - 1, j - 1);\n                pixelMatrix[0][1] = input.getRGB(i - 1, j);\n                pixelMatrix[0][2] = input.getRGB(i - 1, j + 1);\n                pixelMatrix[1][0] = input.getRGB(i, j - 1);\n                pixelMatrix[1][2] = input.getRGB(i, j + 1);\n                pixelMatrix[2][0] = input.getRGB(i + 1, j - 1);\n                pixelMatrix[2][1] = input.getRGB(i + 1, j);\n                pixelMatrix[2][2] = input.getRGB(i + 1, j + 1);\n\n                int edge = (int) convolution(pixelMatrix);\n                int rgb = (edge << 16 | edge << 8 | edge);\n                output[j][i] = rgb;\n            }\n        }\n\n        MatrixSource source = new MatrixSource(output);\n        return source;\n    }",
            'nl': "Expects a height mat as input"}
        , {
            'code': "public static ComplexNumber Add(ComplexNumber z1, ComplexNumber z2) {\r\n        return new ComplexNumber(z1.real + z2.real, z1.imaginary + z2.imaginary);\r\n    }",
            'nl': "Adds two complex numbers."}
        , {
            'code': "public void setOutRGB(IntRange outRGB) {\r\n        this.outRed = outRGB;\r\n        this.outGreen = outRGB;\r\n        this.outBlue = outRGB;\r\n\r\n        CalculateMap(inRed, outRGB, mapRed);\r\n        CalculateMap(inGreen, outRGB, mapGreen);\r\n        CalculateMap(inBlue, outRGB, mapBlue);\r\n    }",
            'nl': "Set RGB output range."}
        , {
            'code': "public Table newRow()\n    {\n        unnest();\n        nest(row = new Block(\"tr\"));\n        if (_defaultRow!=null)\n        {\n            row.setAttributesFrom(_defaultRow);\n            if (_defaultRow.size()>0)\n                row.add(_defaultRow.contents());\n        }\n        cell=null;\n        return this;\n    }",
            'nl': 'Create new table row . Attributes set after this call and before a call to newCell or newHeader are considered row attributes.'}
    ],
    'python': [
        {
            'code': 'def get_meshes_fld(step, var):\n    fld = step.fields[var]\n    if step.geom.twod_xz:\n        xmesh, ymesh = step.geom.x_mesh[:, 0, :], step.geom.z_mesh[:, 0, :]\n        fld = fld[:, 0, :, 0]\n    elif step.geom.cartesian and step.geom.twod_yz:\n        xmesh, ymesh = step.geom.y_mesh[0, :, :], step.geom.z_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    else:  \n        xmesh, ymesh = step.geom.x_mesh[0, :, :], step.geom.y_mesh[0, :, :]\n        fld = fld[0, :, :, 0]\n    return xmesh, ymesh, fld',
            'nl': 'Return scalar field along with coordinates meshes .'}
        , {
            'code': 'def _obtain_token(self):\n        if self.expiration and self.expiration > datetime.datetime.now():\n            return\n        resp = requests.post("{}/1.1/oauth/token".format(API_URL), data={\n            "client_id": self.client_id,\n            "client_secret": self.client_secret,\n            "grant_type": "client_credentials"\n        }).json()\n        if "error" in resp:\n            raise APIError("LibCal Auth Failed: {}, {}".format(resp["error"], resp.get("error_description")))\n        self.expiration = datetime.datetime.now() + datetime.timedelta(seconds=resp["expires_in"])\n        self.token = resp["access_token"]\n        print(self.token)',
            'nl': 'Obtain an auth token from client id and client secret .'}
        , {
            'code': "def get_byte(self, i):\n        value = []\n        for x in range(2):\n            c = next(i)\n            if c.lower() in _HEX:\n                value.append(c)\n            else:  \n                raise SyntaxError('Invalid byte character at %d!' % (i.index - 1))\n        return ''.join(value)",
            'nl': 'Get byte .'}
        , {
            'code': "def get_item_children(item):\r\n    children = [item.child(index) for index in range(item.childCount())]\r\n    for child in children[:]:\r\n        others = get_item_children(child)\r\n        if others is not None:\r\n            children += others\r\n    return sorted(children, key=lambda child: child.line)",
            'nl': "Return a sorted list of all the children items of item ."}
    ]}


class GPT:
    def __init__(self, model):
        # self.openai_key = args.openai_key
        self.model_name = model
        self.temperature = 0
        self.top_p = 1.0

    def ask(self, input, history=[], system_prompt=DEFAULT_SYSTEM_PROMPT):
        client = OpenAI(
            api_key='sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
            base_url="https://api.agicto.cn/v1",
        )
        message = [{"role": "system", "content": system_prompt}]
        for his in history:
            q, a = his
            message.append({"role": "user", "content": q})
            message.append({"role": "assistant", "content": a})

        message.append({"role": "user", "content": input})
        response = client.chat.completions.create(model=self.model_name, messages=message, temperature=self.temperature,
                                                  top_p=self.top_p)
        result = response.choices[0].message.content
        sleep(1)
        return result.strip()


class Claude:
    def __init__(self, model):
        self.model_name = model
        self.temperature = 0
        self.top_p = 1.0

    def ask(self, input, history=[], system_prompt="You are a helpful assistant."):
        api_key = 'sk-8j35NBb9vpfA1kU2QnWWLMitH6bTbi2VjMzZnWFxKbUoYW9V',
        api_base = "https://api.agicto.cn/v1",
        # 拼接整个 prompt
        message = [{"role": "system", "content": system_prompt}]
        for his in history:
            q, a = his
            message.append({"role": "user", "content": q})
            message.append({"role": "assistant", "content": a})
        message.append({"role": "user", "content": input})
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url=api_base
        )
        # 调用 Claude API
        response = client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=0,
            messages=message,
        )
        sleep(1)
        return response.content[0].text.strip()


def generate_summaries_zero_shot(model, code):
    message = model.ask(input=BASIC_PROMPT + code)
    return message


def generate_summaries_few_shot_history(lang, model, code):  # few-shot example as history
    history_prompt = []
    for example in fewshot_example_language_4[lang]:
        ex_code = example['code']
        nl = example['nl']
        history_prompt.append((BASIC_PROMPT + ex_code, nl))
    message = model.ask(input=BASIC_PROMPT + code, history=history_prompt)
    return message


def generate_summaries_chain_of_thought(model, code):
    prompt1 = \
        '''Code:
        {}

        Question：
        1、What is the name of the function?
        2、What are the input parameters that are being accepted by the function?
        3、What is the expected output or return value of the function?
        4、Are there any specific requirements or constraints for using this function?
        5、Does the function have any additional dependencies or external requirements?
        Please Answer the above questions.'''
    prompt2 = 'Let\'s integrate the above information and generate a short comment in one sentence for the function.'
    reply1 = model.ask(input=prompt1.format(code))
    message = model.ask(input=prompt2, history=[(prompt1.format(code), reply1)])
    return message


def generate_summaries_critique(model, code):
    prompt2 = 'Review your previous answer and find problems with your answer.'
    prompt3 = 'Based on the problems you found, improve your answer.'
    reply1 = model.ask(input=BASIC_PROMPT + code)
    reply2 = model.ask(input=prompt2, history=[(BASIC_PROMPT + code, reply1)])
    message = model.ask(input=prompt3, history=[(BASIC_PROMPT + code, reply1), (prompt2, reply2)])
    return message


def generate_summaries_expert_history(model, code):
    expert_prompt = 'For the following instruction, write a high-quality description about the most capable and suitable agent to answer the instruction in second person perspective:\n'

    expert_example = [{'Instruction': 'Make a list of 5 possible effects of deforestation.',
                       'Description': 'You are an environmental scientist with a specialization in the study of ecosystems and their interactions with human activities. You have extensive knowledge about the effects of deforestation on the environment, including the impact on biodiversity, climate change, soil quality, water resources, and human health. Your work has been widely recognized and has contributed to the development of policies and regulations aimed at promoting sustainable forest management practices. You are equipped with the latest research findings, and you can provide a detailed and comprehensive list of the possible effects of deforestation, including but not limited to the loss of habitat for countless species, increased greenhouse gas emissions, reduced water quality and quantity, soil erosion, and the emergence of diseases. Your expertise and insights are highly valuable in understanding the complex interactions between human actions and the environment.'
                       }, {'Instruction': 'Identify a descriptive phrase for an eclipse.',
                           'Description': 'You are an astronomer with a deep understanding of celestial events and phenomena. Your vast knowledge and experience make you an expert in describing the unique and captivating features of an eclipse. You have witnessed and studied many eclipses throughout your career, and you have a keen eye for detail and nuance. Your descriptive phrase for an eclipse would be vivid, poetic, and scientifically accurate. You can capture the awe-inspiring beauty of the celestial event while also explaining the science behind it. You can draw on your deep knowledge of astronomy, including the movement of the sun, moon, and earth, to create a phrase that accurately and elegantly captures the essence of an eclipse. Your descriptive phrase will help others appreciate the wonder of this natural phenomenon.'
                           },
                      {'Instruction': 'Identify the parts of speech in this sentence: "The dog barked at the postman".',
                       'Description': 'You are a linguist, well-versed in the study of language and its structures. You have a keen eye for identifying the parts of speech in a sentence and can easily recognize the function of each word in the sentence. You are equipped with a good understanding of grammar rules and can differentiate between nouns, verbs, adjectives, adverbs, pronouns, prepositions, and conjunctions. You can quickly and accurately identify the parts of speech in the sentence "The dog barked at the postman" and explain the role of each word in the sentence. Your expertise in language and grammar is highly valuable in analyzing and understanding the nuances of communication.'
                       }]
    history_prompt = []
    for example in expert_example:
        ex_ins = example['Instruction']
        ex_des = example['Description']
        history_prompt.append((expert_prompt + ex_ins, ex_des))
    system_prompt = model.ask(expert_prompt + 'Generate a short comment in one sentence for a function.',
                              history=history_prompt, system_prompt='You are a helpful assistant.')
    message = model.ask(input=BASIC_PROMPT + code, system_prompt=system_prompt)
    return message


def SCSL_generate(code, lang, prompt_type, model):
    if model == 'gpt-4-turbo' or model == 'gpt-4':
        m = GPT(model)
    elif model == 'claude-3.7':
        m = GPT('claude-3-7-sonnet-20250219')
    else:
        print('No such model')
        raise Exception('No such model')

    if prompt_type == 'zero':
        return generate_summaries_zero_shot(m, code)
    elif prompt_type == 'few':
        return generate_summaries_few_shot_history(lang, m, code)
    elif prompt_type == 'cot':
        return generate_summaries_chain_of_thought(m, code)
    elif prompt_type == 'critique':
        return generate_summaries_critique(m, code)
    elif prompt_type == 'expert':
        return generate_summaries_expert_history(m, code)
    else:
        print('No such prompt type')
        raise Exception('No such prompt type')


if __name__ == '__main__':
    print(SCSL_generate("def", "python", "zero", "claude-3.7"))
