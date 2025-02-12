import json
import logging
import sys
import time

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant with a deep knowledge of code and software design. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
"""
BASIC_PROMPT = """\
Please score on the code and the corresponding comment given  on a scale of 0-5, and give the rating basis,where a higher score indicates better quality. 
A good comment should: 1) accurately summarize the function of the code; 2) be expressed concisely in one sentence, without burdening the developer with reading; 3) help the developer understand the code quickly.
Output json only and strictly, such as {"score":"","basis":""}\n"""

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def llm_score(code, comment, temperature=0.1, top_p=1.0):
    logger.info('SCORING\n')
    gpt = ChatOpenAI(model='gpt-4-1106-preview',
                     api_key='sk-dK2wyH9nxA3Sf9BEVrRbvmEZyQjUX8wzXMgPCrPtwAgUZ2ux',
                     base_url="https://xiaoai.plus/v1",
                     temperature=temperature,
                     top_p=top_p,
                     response_format={"type": "json_object"}
                     )

    messages = [SystemMessage(content=DEFAULT_SYSTEM_PROMPT),
                HumanMessage(content=BASIC_PROMPT + "\nCode:" + code + "\nComment:" + comment)]
    result = gpt(messages)
    result = result.content.strip()
    logger.info('response:\n' + result)
    return result


if __name__ == '__main__':
    code = "def refresh(self, force=False, _retry=0):\n\t\t\"\"\"\n\t\tCheck if the token is still valid and requests a new if it is not\n\t\tvalid anymore\n\n\t\tCall this method before a call to praw\n\t\tif there might have passed more than one hour\n\n\t\tforce: if true, a new token will be retrieved no matter what\n\t\t\"\"\"\n\t\tif _retry >= 5:\n\t\t\traise ConnectionAbortedError('Reddit is not accessible right now, cannot refresh OAuth2 tokens.')\n\t\tself._check_token_present()\n\n\t\t# We check whether another instance already refreshed the token\n\t\tif time.time() > self._get_value(CONFIGKEY_VALID_UNTIL, float, exception_default=0) - REFRESH_MARGIN:\n\t\t\tself.config.read(self.configfile)\n\n\t\t\tif time.time() < self._get_value(CONFIGKEY_VALID_UNTIL, float, exception_default=0) - REFRESH_MARGIN:\n\t\t\t\tself._log(\"Found new token\")\n\t\t\t\tself.set_access_credentials()\n\n\t\tif force or time.time() > self._get_value(CONFIGKEY_VALID_UNTIL, float, exception_default=0) - REFRESH_MARGIN:\n\t\t\tself._log(\"Refresh Token\")\n\t\t\ttry:\n\t\t\t\tnew_token = self.r.refresh_access_information(self._get_value(CONFIGKEY_REFRESH_TOKEN))\n\t\t\t\tself._change_value(CONFIGKEY_TOKEN, new_token[\"access_token\"])\n\t\t\t\tself._change_value(CONFIGKEY_VALID_UNTIL, time.time() + TOKEN_VALID_DURATION)\n\t\t\t\tself.set_access_credentials()\n\t\t\texcept (praw.errors.OAuthInvalidToken, praw.errors.HTTPException) as e:\n\t\t\t\t# todo check e status code\n\t\t\t\t# self._log('Retrying in 5s.')\n\t\t\t\t# time.sleep(5)\n\t\t\t\t# self.refresh(_retry=_retry + 1)\n\n\t\t\t\tself._log(\"Request new Token (REF)\")\n\t\t\t\tself._get_new_access_information()"
    comment = "The function refreshes OAuth2 tokens for Reddit, handling token expiration and refreshing access credentials when necessary."
    llm_score(code, comment)
