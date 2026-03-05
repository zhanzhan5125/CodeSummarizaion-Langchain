import json
import random

# 文件路径
# file_a = "concise_python.jsonl"
# file_b = "./python/test/main_python_test.jsonl"
#
# selected_file = "./python/test/1800_python_test.jsonl"
# remaining_file = "./python/train/python_train.jsonl"

file_a = "concise_java.jsonl"
file_b = "./java/test/main_java_test.jsonl"

selected_file = "./java/test/1800_java_test.jsonl"
remaining_file = "./java/train/java_train.jsonl"


b_set = set()

with open(file_b, "r", encoding="utf-8") as f:
    for line in f:
        b_set.add(line.strip())

print("B 文件数据量:", len(b_set))

a_data = []
remaining_data = []

with open(file_a, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        if line in b_set:
            continue
        else:
            a_data.append(line)

print("A 中可用数据:", len(a_data))

sample_size = 1800

if len(a_data) < sample_size:
    raise ValueError("可用数据不足 1800 条")

selected = random.sample(a_data, sample_size)

# 剩余数据
selected_set = set(selected)

for line in a_data:
    if line not in selected_set:
        remaining_data.append(line)

with open(selected_file, "w", encoding="utf-8") as f:
    for line in selected:
        f.write(line + "\n")

with open(remaining_file, "w", encoding="utf-8") as f:
    for line in remaining_data:
        f.write(line + "\n")

print("随机数据:", len(selected))
print("剩余数据:", len(remaining_data))