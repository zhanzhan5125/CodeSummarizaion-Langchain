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
log_file_path = os.path.join(os.path.join('./Result/generation/python/codeSummary', 'valid_ai4.txt'))
fh = logging.FileHandler(log_file_path)
logger.addHandler(fh)


def process_string(s):
    if s.count('.') > 1:
        first_dot_index = s.find('.')
        return s[:first_dot_index]
    else:
        return s


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
                comments.append(process_string(' '.join(js['docstring_tokens'])))
                repos.append(js['repo'])
                paths.append(js['path'])
                funcs.append(js['func_name'])
        # 通过zip将多个code和comment配对，生成字典列表
        datas = [{"repo": repo, "path": path, "func": func, "code": code, "comment": comment}
                 for repo, path, func, code, comment in zip(repos, paths, funcs, codes, comments)]
        return datas

    def run_experiment(self, testfile, w_valid, w_origin, count):
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
            origin = result["Origin"]
            # w_valid.writerow([reference, comment, score])
            # w_origin.writerow([reference, origin, score])
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

    test_file = f'./Dataset/{experiment_config["language"]}/test/123_{experiment_config["language"]}_test.jsonl'
    output_file_v = f'./Result/generation/{experiment_config["language"]}/codeSummary/valid_ai4.csv'
    output_file_o = f'./Result/generation/{experiment_config["language"]}/codeSummary/ai4.csv'
    # 运行实验
    cnt = 13
    mode = 'w'
    if cnt > 0: mode = 'a'
    f_v = open(output_file_v, mode, encoding="utf-8", newline='')
    f_o = open(output_file_o, mode, encoding="utf-8", newline='')
    writer_v = csv.writer(f_v)
    writer_o = csv.writer(f_o)

    if cnt == 0:
        writer_v.writerow(['Reference', 'Summary', 'G-eval'])
        writer_o.writerow(['Reference', 'Summary', 'G-eval'])
    runner = ExperimentRunner(experiment_config)
    runner.run_experiment(test_file, writer_v, writer_o, cnt)

    f_v.close()
    f_o.close()
