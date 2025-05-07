import csv
import json
import re
import logging
import string
import warnings
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def process_text(text, lowercase=True, remove_punctuation=True):
    if lowercase:
        text = text.lower()
    if remove_punctuation:
        text = re.sub(r'[^a-zA-Z]', ' ', text)
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
    scores = []
    for code, gen in zip(code_list, generated_list):
        score = llm_score(code, gen, "./g-eval_prompt")["Weighted"]
        scores.append(score)
    return float(np.mean(scores))


def calculate(code_list, reference_list, generated_list, original_reference_list, original_generated_list):
    # 先把 token 列表 join 回字符串
    scores = {"g-eval": g_eval(original_reference_list, original_generated_list),
              "bleu": bleu(reference_list, generated_list),
              "bleu_cn": bleu_cn(reference_list, generated_list),
              "meteor": meteor(reference_list, generated_list),
              "bertscore": bertscore(original_reference_list, original_generated_list, lang='en'),
              "sidescore": sidescore(code_list, original_generated_list)
              }
    scores.update(rouge(reference_list, generated_list))
    return scores


def get_codes(language):
    codes = []
    with open("../Dataset/python/test/123_python_test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            codes.append(remove_comments_and_docstrings(source=js['code'], lang=language))
    return codes


def evaluate_file(data_file, language):
    df = pd.read_csv(data_file)
    if df.shape[1] > 2:
        means = df.iloc[:, 2:].mean(axis=0).round(4)
        # 转成普通字典并返回
        print(means.to_dict())

    else:
        reference_list = df.iloc[:, 0].tolist()
        generated_list = df.iloc[:, 1].tolist()
        rows = df.values.tolist()

        new_header = ['Reference', 'Summary', 'G-eval', 'BLEU', 'BLEU-CN', 'METEOR', 'BERTSCORE', 'SIDESCORE',
                      'ROUGE-1', 'ROUGE-2', 'ROUGE-L']
        new_data = []

        logger.info(f'[Evaluation] ---- [{data_file}]')

        ref_token_list = []
        gen_token_list = []
        code_list = get_codes(language)

        for ref, gen in zip(reference_list, generated_list):
            ref_token = process_text(ref)
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
    # evaluate_file("../Result/generation/python/ASAP/r1.csv", "python")
    # evaluate_file("../Result/generation/python/SCSL/zero_shot.csv", "python")
    # evaluate_file("../Result/generation/python/codeSummary/ai3.csv", "python")
    # evaluate_file("../Result/generation/python/codeSummary/valid_ai3.csv", "python")
    # evaluate_file("../Result/generation/python/codeSummary/r4.csv", "python")
    # evaluate_file("../Result/generation/python/codeSummary/ai4.csv", "python")
    # evaluate_file("../Result/generation/python/codeSummary/valid_ai4.csv", "python")
    df = pd.read_csv("../Result/generation/python/codeSummary/r4.csv")
    reference_list = df.iloc[:, 0].tolist()
    generated_list = df.iloc[:, 1].tolist()
    # code_list = get_codes("python")
    # print(sidescore(code_list, generated_list))
    # print(g_eval(code_list, reference_list))
    # arr =[92.8832, 90.3035, 89.5894, 83.3292, 85.7226, 89.7941, 68.7568, 74.7993, 98.384, 94.2824, 74.937, 98.463, 87.6782, 77.1057, 71.1874, 91.6602, 87.2252, 96.3224, 94.8976, 82.9705, 90.193, 92.4621, 87.0125, 73.0048, -14.3402, 96.0704, 92.9668, 92.0548, 80.3826, 96.6801, 79.636, 97.6539, 63.2311, 90.7386, 93.7163, 82.7121, 80.4137, 79.1252, 98.7603, 91.872, 88.1457, 97.9395, 99.2663, 88.4123, 98.7273, 87.0458, 99.7757, 86.6525, 83.3174, 96.082, 91.5445, 93.1249, 84.6437, 97.1665, 83.2638, 90.9265, 93.0686, 87.266, 95.1366, 82.7637, 97.376, 83.1751, 78.8156, 86.2364, 83.6552, 95.2237, 86.3954, 96.2618, 74.9185, 68.4771, 89.9985, 66.3452, 96.8491, 94.2958, 88.2885, 93.0366, 99.5552, 47.6977, 69.8635, 97.4106, 78.6192, 52.8314, 60.3813, 88.878, 92.7741, 86.8836, 91.0628, 79.9586, 96.7964, 90.2086, 47.2365, 93.7902, 88.3758, 83.2443, 95.4033, 99.652, 79.7881, 88.1066, 98.5706, 94.2159, 84.6403, 89.5101, 97.3392, 93.3893, 88.7775, 91.3882, 77.4199, 91.4101, 62.008, 97.438, 88.0533, 80.6843, 85.5037, 97.7307, 90.63, 93.3085, 78.803, 87.3458, 99.0311, 59.3048, 91.3586, 97.8955]
    # replace_nth_column("../Result/generation/python/codeSummary/r4.csv", arr, 7)
