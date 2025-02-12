import json
import random
import os

# 设定路径
dataset_dir = '../dataset/Dataset'
textset_dir = '../dataset/Testset'
language = ['c', 'python', 'java']

for lang in language:
    filename = lang + '.jsonl'

    # 创建textset目录下的同名文件夹
    output_dir = os.path.join(textset_dir, filename.replace('.jsonl', ''))
    os.makedirs(output_dir, exist_ok=True)

    # 读取c.jsonl文件
    file_path = os.path.join(dataset_dir, filename)
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 随机选择50行数据
    sample_lines = random.sample(lines, 50)

    # 从原文件中删除这50行
    remaining_lines = [line for line in lines if line not in sample_lines]

    # 将剩余数据写回到原c.jsonl文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(remaining_lines)

    # 将选择的50行数据保存到textset目录中的同名文件夹里
    output_file_path = os.path.join(output_dir, filename)
    with open(output_file_path, 'w', encoding='utf-8') as f:
        f.writelines(sample_lines)

    print(f"succeed move 50 random datas from {filename} to {output_file_path}")
