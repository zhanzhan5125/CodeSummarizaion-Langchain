import csv
import json
import logging
import os
import warnings
from agent import AgentRunner
from util.remove_comments import remove_comments_and_docstrings

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
log_file_path = os.path.join(os.path.join('./Result/generation/python/codeSummary', 'test.txt'))
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)


class ExperimentRunner:
    def __init__(self, config):
        self.config = config
        self.language = config['language']

    def get_data(self, testfile):
        repos = []
        paths = []
        funcs = []
        codes = []
        comments = []
        with open(testfile, "r", encoding="utf-8") as f:
            print("opening file ", testfile)
            for line in f:
                line = line.strip()
                js = json.loads(line)
                codes.append(remove_comments_and_docstrings(source=js['code'], lang=self.language))
                comments.append(' '.join(js['docstring_tokens']))
                repos.append(js['repo'])
                paths.append(js['path'])
                funcs.append(js['func_name'])
        # 通过zip将多个code和comment配对，生成字典列表
        datas = [{"repo": repo, "path": path, "func": func, "code": code, "comment": comment}
                 for repo, path, func, code, comment in zip(repos, paths, funcs, codes, comments)]
        return datas

    def run_experiment(self, testfile, w, count):
        llm_set = self.config
        agent = AgentRunner(llm_set)
        datas = self.get_data(testfile)
        for idx, data in enumerate(datas, start=1):
            if idx < count: continue
            logger.info(f"===============================[ {idx} ]===============================")
            logger.info(data['code'])
            result = agent.run(data)
            reference = data['comment']
            comment = result["Comment"]
            score = result["Score"]
            # w.writerow([reference, comment, score])
            logger.info("\n")
            break


if __name__ == "__main__":
    # 示例实验配置
    experiment_config = {
        "model": "gpt-4",
        "prompt_type": "codeSummary",
        "rag": "embedding",
        "temperature": 0.1,
        "top_p": 1.0,
        "max_iter": 3,
        "language": "python"
    }

    test_file = f'./Dataset/{experiment_config["language"]}/test/187_{experiment_config["language"]}_test.jsonl'
    output_file = f'./Result/generation/{experiment_config["language"]}/ASAP/test.csv'
    # 运行实验
    cnt = 1
    mode = 'w'
    if cnt > 0: mode = 'a'
    f = open(output_file, mode, encoding="utf-8", newline='')
    writer = csv.writer(f)

    if cnt == 0: writer.writerow(['Reference', 'Summary', 'G-eval'])
    runner = ExperimentRunner(experiment_config)
    runner.run_experiment(test_file, writer, cnt)

    f.close()
