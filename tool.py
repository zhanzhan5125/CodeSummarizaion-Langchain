from langchain_core.tools import StructuredTool
from rag import CodeRetriever
from validation import llm_score
from extract_context import get_file_content


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
                name="extract_context",
                func=self.extract_context,
                description="Extract the context from source file"
            )
        ]
    def retrieve_samples(self, data: dict, method: str):
        """Retrieve similar code-comment pairs from database"""
        return self.retriever.query_samples(data, method)

    def valid_comment(self, code: str, comment: str, model_v: str, prompt_file: str, s_min:int, s_max:int):
        """Evaluate the quality of generated comment"""
        return llm_score(code, comment, model_v, prompt_file, s_min, s_max)

    def extract_context(self, lang:str, repo: str, path: str, sha: str = None):
        """Extract the context from source file"""
        return get_file_content(lang, repo, path, sha)


