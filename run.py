import argparse
import copy
import csv
import json
import logging
import sys
import warnings

from tqdm import tqdm
from rag import query_n
from model import GPT, CLAUDE
import random
import os
from util.remove_comments import remove_comments_and_docstrings
from score import llm_score

warnings.filterwarnings('ignore')

# args.logger.info('zero-shot prompt...')

testset_dir = './dataset/Testset'
language = ['c', 'java', 'python']


def record_feedback(model, score, code, basis, summary, feedback_list):
    while score < 4:
        print('start feedback........')
        summary = model.feedback(code, summary, score, basis)
        score_json = json.loads(llm_score(code, summary))
        score = int(score_json['score'])
        basis = score_json['basis']
        # 将每次迭代的 score 和 basis 存入列表
        feedback_list.append((score, basis, summary))
    return summary


def zero_shot_prompting(args, model, datas, output_file, cnt=0):
    args.logger.info('zero-shot prompt...')
    print(f'***********************{args.language} ZERO_SHOT ***************************')
    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)
    # 使用 enumerate 获取索引、code 和 comment
    for idx, data in enumerate(datas):
        if idx < cnt: continue
        code = data["code"]
        comment = data["comment"]
        print(f'current idx:{idx}\n')
        print('generating original summary........')

        original_answer = model.ask(code, 0)

        print('start scoring........')
        score_json = json.loads(llm_score(code, original_answer))

        score = int(score_json['score'])
        basis = score_json['basis']
        feedback_list = [(score, basis, "#")]
        summary = record_feedback(model, score, code, basis, original_answer, feedback_list)
        # 将 summary 和所有的 score-basis 对写入文件
        if summary == original_answer:
            summary = "as same as original answer"
        row = [idx, original_answer, summary, len(feedback_list)-1, feedback_list]
        writer.writerow(row)

    print(f'***********************{args.language} DONE ZERO_SHOT ***************************')

    f.close()


def few_shot_prompting_n(args, model, datas, output_file, cnt=0):
    args.logger.info('few-shots prompt...')
    print(f'***********************{args.language} FEW_SHOTS ***************************')
    lang = args.language
    n = args.few_shots
    f = open(output_file, args.mode, encoding="utf-8", newline='')
    writer = csv.writer(f)

    for idx, data in enumerate(datas):
        if idx < cnt: continue
        code = data["code"]
        comment = data["comment"]
        print('current idx:', idx)
        print('generating original summary........')
        samples = similar_samples_n(code, lang, n)

        original_answer = model.ask(code, 1, samples)

        print('start scoring........')
        score_json = json.loads(llm_score(code, original_answer))

        score = int(score_json['score'])
        basis = score_json['basis']
        feedback_list = [(score, basis, "#")]
        summary = record_feedback(model, score, code, basis, original_answer, feedback_list)
        # 将 summary 和所有的 score-basis 对写入文件
        if summary == original_answer:
            summary = "as same as original answer"
        row = [idx, original_answer, summary, len(feedback_list) - 1, feedback_list]
        writer.writerow(row)

    print(f'***********************{args.language} DONE FEW_SHOTS ***************************')
    f.close()


def similar_samples_n(code, lang, n):
    similar_samples = query_n(code, lang, n)
    return similar_samples


def get_data(args):
    lang = args.language
    testfile = testset_dir + '/' + lang + '/' + lang + '.jsonl'
    codes = []
    comments = []
    length = 0
    with open(testfile, "r", encoding="utf-8") as f:
        print("opening file ", testfile)
        for line in f:
            length += 1
            line = line.strip()
            js = json.loads(line)
            if lang == 'c':
                codes.append(remove_comments_and_docstrings(source=js['function'], lang=lang))
                comment = js['summary']
                if comment.endswith('.'):
                    comment = comment[:-1]
                comment = comment + ' .'
                comments.append(comment)
            else:
                codes.append(remove_comments_and_docstrings(source=js['code'], lang=lang))
                comments.append(' '.join(js['docstring_tokens']))
    # 通过zip将多个code和comment配对，生成字典列表
    datas = [{"code": code, "comment": comment} for code, comment in zip(codes, comments)]
    return datas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default="c", type=str)
    parser.add_argument("--model", default="gpt-3.5", type=str)
    parser.add_argument("--count", default=0, type=int, help="continue from sample `count`")
    parser.add_argument("--temperature", default=0.1, type=float)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--few_shots", default=2, type=int)
    parser.add_argument("--log_filename", default='log.txt', type=str)
    args = parser.parse_args()

    # ouput summary directory
    dir = './results/summaries/{}/{}/{}/{}/'.format(args.language, args.model, args.temperature, args.top_p)
    if not os.path.exists(dir):
        os.makedirs(dir)

    # logger
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    args.logger = logging.getLogger(__name__)
    log_file_path = os.path.join(os.path.join(dir, args.log_filename))
    fh = logging.FileHandler(log_file_path)
    args.logger.addHandler(fh)  # add the handlers to the logger
    args.logger.info("Training/evaluation parameters %s", args)
    args.logger.info("\n")

    MODEL_NAME_OR_PATH = {'gpt-4': 'gpt-4-1106-preview',
                          'gpt-3.5': 'gpt-3.5-turbo',
                          # 'claude-3.5': 'claude-3.5'
                          }
    args.model_name_or_path = MODEL_NAME_OR_PATH[args.model]

    if args.model == 'gpt-4':
        llm = GPT(args=args)
    elif args.model == 'gpt-3.5':
        llm = GPT(args=args)
    else:
        print('Model not found!')
        sys.exit(1)

    if args.count > 0:
        args.mode = 'a'
    else:
        args.mode = 'w'

    datas = get_data(args)
    # zero_shot_prompting(args, llm, datas, dir + 'zero_shot.csv', args.count)
    few_shot_prompting_n(args, llm, datas, dir + 'few_shot.csv', args.count)

if __name__ == '__main__':
    main()