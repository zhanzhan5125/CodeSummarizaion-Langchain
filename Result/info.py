import csv
import json

import pandas as pd


def low_score(csv_file, score, is_delete=False):
    df = pd.read_csv(csv_file, header=None, encoding='latin-1')
    mask = df.iloc[:, 0].astype(int) <= score
    row_indices = (df[mask].index + 1).tolist()  # +1 转换为自然行号
    if is_delete:
        df = df.drop(df[mask].index)
        df.to_csv(csv_file, index=False, header=False)
    return row_indices


def delete_multiple_jsonl_lines(filepath, line_numbers_to_delete):
    """
    删除 JSONL 文件中从 1 开始计数的多行数据。
    :param filepath: JSONL 文件路径
    :param line_numbers_to_delete: 要删除的行号列表（从1开始计数）
    """
    if not line_numbers_to_delete:
        return

    # 将行号转为 set 并转换为从 0 开始的索引
    target_indices = set([i - 1 for i in line_numbers_to_delete if i >= 1])

    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 检查是否越界
    max_line = len(lines)
    for i in target_indices:
        if i >= max_line:
            raise IndexError(f"文件只有 {max_line} 行，不能删除第 {i + 1} 行")

    # 删除指定行
    new_lines = [line for idx, line in enumerate(lines) if idx not in target_indices]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"已成功删除 {len(target_indices)} 行：{sorted(line_numbers_to_delete)}")




def calculate_average(csv_file, column_n, ignore_indices=None):
    if ignore_indices is None:
        ignore_indices = []
    df = pd.read_csv(csv_file, header=None, encoding='latin-1')
    ignore_indexes = [x - 1 for x in ignore_indices]
    filtered_df = df.drop(ignore_indexes)
    return filtered_df.iloc[:, column_n].astype(float).mean()


def delete_result(csv_file, l):
    df = pd.read_csv(csv_file, header=None, encoding='latin-1')
    df = df.drop(df.index[l])
    df.to_csv(csv_file, index=False, header=False)


def find_validation(examples, file_json, file_csv):
    for example in examples:
        with open(file_json, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                if example['data']['func'] == js['func_name']:
                    df = pd.read_csv(file_csv, header=None,encoding='latin-1')
                    example['Validation'] = df.loc[idx, 2]
        return None


def extract_origin(input_file,csv_file):
    # 要匹配的前缀
    delimiter = '----------------Refine 2----------------'  # 标记行
    prefix = '*** Refine comment ***:'  # 要匹配的前缀

    comments = []
    # 读取所有行
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.strip() == delimiter and i + 1 < len(lines):
                comments.append(lines[i+1][len(prefix):].strip())

    # 读取已存在的 CSV
    with open(csv_file, 'r', encoding='utf-8', newline='') as f:
        reader = list(csv.reader(f))
        header = reader[0]
        data_rows = reader[1:]

    # 检查提取数量与 CSV 行数
    if len(comments) != len(data_rows):
        print(f"警告: 提取的评论数量 {len(comments)} 与 CSV 数据行数 {len(data_rows)} 不一致，按最小值填充。")
    min_len = min(len(comments), len(data_rows))

    # 更新第二列（索引 1）
    for i in range(min_len):
        data_rows[i][1] = comments[i]

    # 写回 CSV
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data_rows)

    print(f"已将前 {min_len} 条评论写入 '{csv_file}' 第二列。")


if __name__ == '__main__':
    # low = [127,129,130,133,136,137,138,143,146,148,149,197]
    # delete_multiple_jsonl_lines("../Dataset/java/test/main_java_test.jsonl", low)
    low = low_score("../Result/trainset_validation/2000_java_valid.csv", 1, True)
    print(len(low))
    delete_multiple_jsonl_lines("../Dataset/java/train/2000_java_train.jsonl", low)
    # 31, 32, 34, 35, 36, 38, 39, 40, 41, 45
    # 2,3,4,7,8,10,11,12
    # testfile = "../Dataset/python/test/222_python_test.jsonl"
    # with open(testfile, "r", encoding="utf-8") as f:
    #     print("opening file ", testfile)
    #     cnt = 0
    #     for line in f:
    #         cnt += 1
    #         if cnt < 0: continue
    #         print(f"=============={cnt}================")
    #         line = line.strip()
    #         js = json.loads(line)
    #         comment = ' '.join(js['docstring_tokens'])
    #         print(comment)
    #         if cnt >= 15: break

