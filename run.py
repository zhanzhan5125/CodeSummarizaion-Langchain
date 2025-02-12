import json
import logging
import warnings
from agent import AgentRunner
from util.remove_comments import remove_comments_and_docstrings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')


class ExperimentRunner:
    def __init__(self, config):
        """ config示例：
        {
            "model": "gpt-4",
            "prompt_type": "few_shot",
            "temperature": 0.1,
            "max_iter": 3,
            "language": c

        }
        """
        self.config = config
        self.language = config['language']

    def get_data(self, testfile):
        codes = []
        comments = []
        length = 0
        with open(testfile, "r", encoding="utf-8") as f:
            print("opening file ", testfile)
            for line in f:
                length += 1
                line = line.strip()
                js = json.loads(line)
                if self.language == 'c':
                    codes.append(remove_comments_and_docstrings(source=js['function'], lang=self.language))
                    comment = js['summary']
                    if comment.endswith('.'):
                        comment = comment[:-1]
                    comment = comment + ' .'
                    comments.append(comment)
                else:
                    codes.append(remove_comments_and_docstrings(source=js['code'], lang=self.language))
                    comments.append(' '.join(js['docstring_tokens']))
        # 通过zip将多个code和comment配对，生成字典列表
        datas = [{"code": code, "comment": comment} for code, comment in zip(codes, comments)]
        return datas

    def run_experiment(self, test_file):
        results = []
        llm_set = self.config
        agent = AgentRunner(llm_set)

        datas = self.get_data(test_file)

        for idx, data in enumerate(datas):
            logger.info(f"#############Processing data {idx}")
            result = agent.run(data["code"])
            results.append({
                "original_code": data["code"],
                "generated_comment": result["final_output"],
                "iterations": result["metadata"]["iterations"]
            })
        return results


if __name__ == "__main__":
    # 示例实验配置
    experiment_config = {
        "model": "gpt-4",
        "prompt_type": "few_shot",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_iter": 3,
        "language": "c"
    }

    test_file = './dataset/Testset' + '/' + experiment_config["language"] + '/' + experiment_config["language"] + '.jsonl'
    # 运行实验
    runner = ExperimentRunner(experiment_config)
    results = runner.run_experiment(test_file)
