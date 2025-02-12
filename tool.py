import json
from langchain_core.tools import StructuredTool
from rag1 import CodeRetriever
from score import llm_score


class CodeTools:
    def __init__(self):
        self.retriever = CodeRetriever(3)
        self.tools = [
            StructuredTool.from_function(
                name="retrieve_similar_code",
                func=self.retrieve_samples,
                description="Retrieve similar code-comment pairs from database"
            ),
            StructuredTool.from_function(
                name="evaluate_comment",
                func=self.evaluate_comment,
                description="Evaluate the quality of generated comment"
            ),
        ]

    def retrieve_samples(self, code:str, lang:str):
        """Retrieve similar code-comment pairs from database"""
        return self.retriever.query_samples(code, lang)

    def evaluate_comment(self, code:str,comment:str):
        """Evaluate the quality of generated comment"""
        return json.loads(llm_score(code, comment))
