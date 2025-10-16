import argparse
import csv
import json
import logging
import os
import time
import warnings
from agent import AgentRunner
from util.remove_comments import remove_comments_and_docstrings


class ExperimentRunner:
    def __init__(self, args):
        self.config = vars(args)
        self.language = args.language
        self.logger = args.logger

    def get_data(self, testfile):
        repos = []
        paths = []
        shas = []
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
                shas.append(js['sha'])
                funcs.append(js['func_name'])
        # 通过zip将多个code和comment配对，生成字典列表
        datas = [{"repo": repo, "path": path, "sha": sha, "func": func, "code": code, "comment": comment}
                 for repo, path, sha, func, code, comment in zip(repos, paths, shas, funcs, codes, comments)]
        return datas

    def run_experiment(self, testfile, w_valid, w_novalid, count):
        llm_set = self.config
        agent = AgentRunner(llm_set, self.logger)
        datas = self.get_data(testfile)
        for idx, data in enumerate(datas, start=1):
            if idx < count: continue
            self.logger.info(f"===============================[ {idx} ]===============================")
            self.logger.info("\n" + data['code'])
            start_time = time.time()  # 记录开始时间
            result = agent.run(data)
            end_time = time.time()  # 记录结束时间
            elapsed_time = end_time - start_time  # 计算用时
            reference = data['comment']
            comment = result["Comment"]
            original_comment = result["Origin"]
            # w_valid.writerow([reference, comment, elapsed_time])
            # w_novalid.writerow([reference, original_comment])
            # print(reference)
            self.logger.info("\n")
            # if idx % 10 ==0:
            #     break
            break


if __name__ == "__main__":
    # 示例实验配置
    parser = argparse.ArgumentParser()
    parser.add_argument("--rq", default="RQ1", type=str)
    parser.add_argument("--language", default="python", type=str)
    parser.add_argument("--model", default="gpt-4-turbo", type=str)
    parser.add_argument("--model_v", default="gpt-4o", type=str)
    parser.add_argument("--count", default=0, type=int, help="continue from sample `count`")
    parser.add_argument("--temperature", default=0, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--max_iter", default=3, type=int)
    parser.add_argument("--max_score", default=10, type=int)
    parser.add_argument("--min_score", default=1, type=int)
    parser.add_argument("--method", default="codeSummary", type=str)
    parser.add_argument("--prompt_type", default="no", type=str)
    parser.add_argument("--rag", default="embedding", type=str)
    parser.add_argument("--log_filename", default='log.txt', type=str)
    args = parser.parse_args()

    # ouput directory
    dir = ''
    if args.rq == "RQ1":
        if args.method == "codeSummary":
            dir = './Result/RQ1/{}/{}/{}/{}/'.format(args.language, args.method, args.model, args.model_v)
        elif args.method == "SCSL":
            dir = './Result/RQ1/{}/{}/{}/{}/'.format(args.language, args.method, args.model, args.prompt_type)
        else:
            dir = './Result/RQ1/{}/{}/{}/'.format(args.language, args.method, args.model)
    elif args.rq == "RQ2":
        dir = './Result/RQ2/{}/{}/{}/{}/{}'.format(args.language, args.method, args.model, args.model_v, args.max_iter)
    elif args.rq == "RQ3":
        dir = './Result/RQ3/{}/{}/{}/{}/{}'.format(args.language, args.method, args.model, args.model_v,
                                                   str(args.min_score) + "_" + str(args.max_score))

    if not os.path.exists(dir):
        os.makedirs(dir)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    warnings.filterwarnings('ignore')
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))
    fh = logging.FileHandler(log_file_path)
    args.logger.addHandler(fh)
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")

    test_file = f'./Dataset/{args.language}/test/main_{args.language}_test.jsonl'
    output_file_v = os.path.join(os.path.join(dir, "valid.csv"))
    output_file_o = os.path.join(os.path.join(dir, "no_valid.csv"))
    # 运行实验
    # args.count = 112
    mode = 'w'
    if args.count > 0: mode = 'a'
    f_v = open(output_file_v, mode, encoding="utf-8", newline='')
    f_o = open(output_file_o, mode, encoding="utf-8", newline='')
    writer_v = csv.writer(f_v)
    writer_o = csv.writer(f_o)

    if args.count == 0:
        writer_v.writerow(['Reference', 'Summary', 'Time'])
        writer_o.writerow(['Reference', 'Summary'])
    runner = ExperimentRunner(args)
    runner.run_experiment(test_file, writer_v, writer_o, args.count)

    f_v.close()
    f_o.close()
