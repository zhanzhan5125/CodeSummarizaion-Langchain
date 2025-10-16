import csv
import json
import re
import logging
import string
import warnings
from nltk.translate.bleu_score import *
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from metric.codenn_bleu import codenn_smooth_bleu
from bert_score import score as bert_score
from validation import llm_score
from SIDE import side_score
from util.remove_comments import remove_comments_and_docstrings


def process_text(text, lowercase=True, remove_punctuation=True):
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        # 去除所有标点符号
        text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)

    # 在字母和数字之间添加空格（双向）
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

    tokens = word_tokenize(text)
    return tokens


def bleu_cn(reference_list, generated_list):
    print('Calculating BLEU-CN...')
    ref_str_list = []
    gen_str_list = []
    for reference, gen in zip(reference_list, generated_list):
        ref_str_list.append([" ".join([str(token_id) for token_id in reference[0]])])
        gen_str_list.append(" ".join([str(token_id) for token_id in gen]))
    score = codenn_smooth_bleu(ref_str_list, gen_str_list)[0]
    return round(score, 4)


def bleu(reference_list, generated_list):
    print('Calculating BLEU...')
    smoothie = SmoothingFunction().method4
    scores = []
    print(reference_list, generated_list)
    for ref, gen in zip(reference_list, generated_list):
        score = sentence_bleu(ref, gen, weights=[0.25, 0.25, 0.25, 0.25], smoothing_function=smoothie)
        scores.append(score)
    score = np.mean(scores)
    return float(round(score * 100, 4))


def rouge(reference_list, generated_list):
    print('Calculating ROUGE...')
    ref_str_list = []
    gen_str_list = []
    for reference, gen in zip(reference_list, generated_list):
        ref_str_list.append(" ".join([str(token_id) for token_id in reference[0]]))
        gen_str_list.append(" ".join([str(token_id) for token_id in gen]))
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False, split_summaries=True)
    scores = []
    types = {}
    for rouge_type in ['rouge1', 'rouge2', 'rougeL']:
        for ref, gen in zip(ref_str_list, gen_str_list):
            score = scorer.score(ref, gen)[rouge_type]
            scores.append(round(score.fmeasure, 4))
        types[rouge_type] = float(round(np.mean(scores) * 100, 4))
        scores = []
    return types


def meteor(reference_list, generated_list):
    print('Calculating METEOR...')
    scores = [meteor_score(ref, gen) for ref, gen in zip(reference_list, generated_list)]
    score = np.mean(scores)
    return float(round(score * 100, 4))


def bertscore(reference_list, generated_list, lang='en'):
    """
    对一批 reference_list（List[str]）和 generated_list（List[str]）计算 BERTScore。
    返回平均 F1 分数（乘以100并保留四位小数）。
    """
    # bert_score 返回 P, R, F 向量；我们取 F 向量的平均
    print('Calculating BERTSCORE...')
    P, R, F = bert_score(
        cands=generated_list,
        refs=reference_list,
        lang=lang,
        verbose=True,
        model_type='bert-large-uncased'
    )
    # .tolist() 转为 Python list，再取平均并放大 100
    avg_f1 = float(F.mean().item() * 100)
    return round(avg_f1, 4)


def sidescore(code_list, generated_list):
    print('Calculating SIDESCORE...')
    scores = []
    for code, gen in zip(code_list, generated_list):
        score = side_score(
            code,
            gen,
            model_path="./SIDE-Models/baseline/103080"
        )
        scores.append(score)
    print(scores)
    return float(round(np.mean(scores), 4))


def g_eval(code_list, generated_list):
    print('Calculating G_EVAL...')
    eval_types = ['Generalization']
    scores = {etype: [] for etype in eval_types}

    for code, gen in zip(code_list, generated_list):
        llm = llm_score(code, gen, 'gpt-4o', "./e", 1,5,True)
        for etype in eval_types:
            scores[etype].append(float(llm[etype]))
    g_score = {etype: float(round(np.mean(scores[etype]), 4)) for etype in eval_types}
    return g_score


def calculate(code_list, reference_list, generated_list, original_reference_list, original_generated_list):
    # 先把 token 列表 join 回字符串
    scores = {
              "bleu": bleu(reference_list, generated_list),
              "bleu_cn": bleu_cn(reference_list, generated_list),
              "meteor": meteor(reference_list, generated_list),
              "bertscore": bertscore(original_reference_list, original_generated_list, lang='en'),
              "sidescore": sidescore(code_list, original_generated_list),
              "rougeL": rouge(reference_list, generated_list)["rougeL"]
              }
    scores.update(g_eval(original_reference_list, original_generated_list))
    print(scores)
    return scores


def get_codes(language):
    codes = []
    with open(f"../Dataset/{language}/test/main_{language}_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            codes.append(remove_comments_and_docstrings(source=js['code'], lang=language))
    return codes


def evaluate_reference(reference_file):
    df = pd.read_csv(reference_file)
    if df.shape[1] > 2:
        means = df.iloc[:, 2:].mean(axis=0).round(4)
        # 转成普通字典并返回
        print(means.to_dict())
    else:
        code_list = df.iloc[:, 0].tolist()
        reference_list = df.iloc[:, 1].tolist()
        rows = df.values.tolist()

        new_header = ['Reference', 'Summary', 'SIDESCORE', 'G-eval', 'OverAll']
        # 'Correctness', 'Conciseness', 'Completeness', 'Cohesiveness'
        new_data = []

        for code, refer in zip(code_list, reference_list):
            scores = {
                "sidescore": sidescore([code], [refer]),
            }
            scores.update(g_eval([code], [refer]))
            new_data.append(list(scores.values()))

        o = open(reference_file, 'w', encoding="utf-8", newline='')
        writer = csv.writer(o)
        new_rows = [
            row + new_data[i] for i, row in enumerate(rows)  # 每行追加新数据
        ]
        writer.writerow(new_header)  # 写入新表头
        writer.writerows(new_rows)  # 写入合并后的数据
        o.close()


def evaluate_file(data_file, language):
    df = pd.read_csv(data_file)
    if df.shape[1] > 3:
        means = df.iloc[:, 2:].mean(axis=0).round(4)
        # 转成普通字典并返回
        print(means.to_dict())
    else:
        reference_list = df.iloc[:, 0].tolist()
        generated_list = df.iloc[:, 1].tolist()
        rows = df.values.tolist()

        new_header = ['Reference', 'Summary', 'BLEU', 'BLEU-CN', 'METEOR', 'BERTSCORE', 'SIDESCORE', 'ROUGE-L',
                      'G-eval', 'OverAll']
        # 'Correctness', 'Conciseness', 'Completeness', 'Cohesiveness'
        new_data = []

        ref_token_list = []
        gen_token_list = []
        code_list = get_codes(language)

        for ref, gen in zip(reference_list, generated_list):
            ref_token = process_text(ref)
            print(gen)
            print(type(gen))
            gen_token = process_text(gen)
            ref_token_list.append([ref_token])
            gen_token_list.append(gen_token)

        for code, ref, gen, o_ref, o_gen in zip(code_list, ref_token_list, gen_token_list, reference_list,
                                                generated_list):
            scores = calculate([code], [ref], [gen], [o_ref], [o_gen])
            new_data.append(list(scores.values()))
        o = open(data_file, 'w', encoding="utf-8", newline='')
        writer = csv.writer(o)
        new_rows = [
            row + new_data[i] for i, row in enumerate(rows)  # 每行追加新数据
        ]
        writer.writerow(new_header)  # 写入新表头
        writer.writerows(new_rows)  # 写入合并后的数据
        o.close()


def replace_nth_column(csv_path: str, new_col: list, n: int, output_path: str = None) -> None:
    df = pd.read_csv(csv_path)  # :contentReference[oaicite:0]{index=0}
    if not (0 <= n < df.shape[1]):
        raise IndexError(f"列索引 n={n} 超出范围，应在 [0, {df.shape[1] - 1}] 之间。")
    if len(new_col) != len(df):
        raise ValueError(f"new_col 长度 {len(new_col)} 与 CSV 行数 {len(df)} 不匹配。")
    df.iloc[:, n] = new_col  # :contentReference[oaicite:1]{index=1}
    dest = output_path if output_path else csv_path
    df.to_csv(dest, index=False)


if __name__ == '__main__':
    # evaluate_file("../Result/generation/java/codeSummary/gpt-4-turbo/gpt-4o/valid.csv", "java")
    # evaluate_file("../Result/generation/java/codeSummary/gpt-4-turbo/gpt-4o/no_valid.csv", "java")
    # evaluate_file("../Result/generation/java/ASAP/gpt-4-turbo/valid.csv", "java")
    # evaluate_file("../Result/RQ1/python/codeSummary/claude-3.7/gpt-4o/no_rag_valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/gpt-4o/no_rag_valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/gpt-4o/suggesting_valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/gpt-4o/no_valid.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/1/valid.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/no_valid.csv","python")
    # evaluate_file("../Result/RQ3/python/codeSummary/gpt-4-turbo/gpt-4o/1_5/valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/claude-3.7/gpt-4-turbo/valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/gpt-4o/valid.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/refine_level_3.csv","python")
    # evaluate_file("../Result/RQ3/python/codeSummary/gpt-4-turbo/gpt-4o/1_25/valid.csv","python")
    # evaluate_file("../Result/RQ3/python/codeSummary/gpt-4-turbo/gpt-4o/1_50/valid.csv","python")
    # evaluate_file("../Result/RQ3/python/codeSummary/gpt-4-turbo/gpt-4o/1_75/valid.csv","python")
    # evaluate_file("../Result/RQ3/python/codeSummary/gpt-4-turbo/gpt-4o/1_100/valid.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/refine_level_1.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/refine_level_2.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/refine_level_3.csv","python")
    # evaluate_file("../Result/RQ2/python/codeSummary/gpt-4-turbo/gpt-4o/5/refine_level_4.csv","python")

    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/claude-3.7/valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/claude-3.7/gpt-4-turbo/valid.csv","python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/gpt-4-turbo/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4o/gpt-4o/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4o/gpt-4-turbo/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/o3-mini/gpt-4o/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4-turbo/o3-mini/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/o3-mini/o3-mini/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/gpt-4o/o3-mini/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/codeSummary/o3-mini/gpt-4-turbo/valid.csv", "python")
    # evaluate_file("../Result/RQ1/python/SCSL/gpt-4-turbo/expert/valid.csv","python")
    # evaluate_file("../Result/RQ1/python/SCSL/gpt-4-turbo/critique/valid.csv","python")
    # evaluate_file("../Result/RQ1/java/SCSL/gpt-4-turbo/expert/valid.csv","java")
    # evaluate_file("../Result/RQ1/java/SCSL/gpt-4-turbo/critique/valid.csv","java")
    # evaluate_file("../Result/RQ1/java/SCSL/gpt-4-turbo/cot/valid.csv", "java")
    # evaluate_file("../Result/RQ1/java/SCSL/gpt-4-turbo/few/valid.csv", "java")
    # evaluate_file("../Result/RQ1/java/SCSL/gpt-4-turbo/zero/valid.csv", "java")


    evaluate_file("test.csv", 'python')
    # evaluate_file("../Result/generation/java/SCSL/gpt-4-turbo/zero/valid.csv", "java")
