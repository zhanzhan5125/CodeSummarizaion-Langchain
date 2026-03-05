import csv
import json
import os

from util.remove_comments import remove_comments_and_docstrings

dataset = "./1800_python_test.jsonl"
output_file_v = './1800_reference.csv'
f_v = open(output_file_v, 'w', encoding="utf-8", newline='')
writer_v = csv.writer(f_v)
writer_v.writerow(['Code', 'Reference Comment'])

with open(dataset, "r", encoding="utf-8") as f:
    print("opening file ", dataset)
    for line in f:
        line = line.strip()
        js = json.loads(line)
        code = remove_comments_and_docstrings(source=js['code'], lang='python')
        comment = ' '.join(js['cleaned_docstring_tokens'])
        writer_v.writerow([code, comment])

f_v.close()

