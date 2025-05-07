from langchain_core.tools import StructuredTool
from rag import CodeRetriever
from validation import llm_score
from simplification import simplify


class CodeTools:
    def __init__(self, k, lang):
        self.retriever = CodeRetriever(k, lang)
        self.tools = [
            StructuredTool.from_function(
                name="retrieve_similar_code",
                func=self.retrieve_samples,
                description="Retrieve similar code-comment pairs from database"
            ),
            StructuredTool.from_function(
                name="valid_comment",
                func=self.valid_comment,
                description="Evaluate the quality of generated comment"
            ),
            StructuredTool.from_function(
                name="simplify_code",
                func=self.simplify_code,
                description="Simplify the comment"
            ),
        ]
    def retrieve_samples(self, data: dict, method: str):
        """Retrieve similar code-comment pairs from database"""
        return self.retriever.query_samples(data, method)

    def valid_comment(self, code: str, comment: str, prompt_file: str):
        """Evaluate the quality of generated comment"""
        return llm_score(code, comment, prompt_file)

    def simplify_code(self, comment:str):
        """Simplify the comment"""
        return simplify(comment)

