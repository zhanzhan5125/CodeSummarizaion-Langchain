import csv
import re
import string
import warnings
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')


def process_text_to_comment(text: str) -> str:
    # 按照原 logic 对单段文本提取首句或注释
    paragraph = text.strip().split('\n')
    paragraph = [p for p in paragraph if p.strip()]

    # 如果是 /* ... */ 格式
    if paragraph and paragraph[0].strip().startswith('/*'):
        first = paragraph[0].strip()
        # 多行注释
        if len(first) <= 3:
            final_lines = []
            for line in paragraph:
                s = line.strip()
                if s.startswith('*/'):
                    break
                if s.startswith('*') or s.startswith('/*'):
                    final_lines.append(s.lstrip('/*').lstrip('*').strip())
                else:
                    final_lines.append(s)
            new_paragraph = ' '.join(final_lines)
            sentences = [s for s in new_paragraph.split('. ') if len(s.strip()) > 1]
            comment = sentences[0] if sentences else new_paragraph
        else:
            # 单行注释
            comment = first.lstrip('/*').rstrip('*/').strip()
    else:
        # 普通文本或 // 注释
        first_para = paragraph[0] if paragraph else ''
        sentences = [s for s in first_para.split('. ') if len(s.strip()) > 1]
        comment = sentences[0] if sentences else first_para

        # 去除首尾特殊符号
        comment = comment.strip()
        if comment and comment[0] in '“`-#*"':
            comment = comment[1:].strip()
        if comment.endswith('"') or comment.endswith('*/'):
            comment = re.sub(r'"$|\*/$', '', comment).strip()
        if comment.startswith('//') or comment.startswith('""'):
            comment = comment.lstrip('/"')
        if comment.endswith('.'):
            comment = comment[:-1].strip()
        comment = comment.replace('\t', ' ')
    return comment + ' .'


def beautify_csv(input_csv: str):
    """
    读取 CSV 文件，对第二列（Summary）应用 beautify 逻辑并覆盖原列。
    会在原文件写回。
    """
    tmp_file = input_csv + '.tmp'
    with open(input_csv, 'r', encoding='utf-8', newline='') as rf, \
         open(tmp_file, 'w', encoding='utf-8', newline='') as wf:
        reader = csv.reader(rf)
        writer = csv.writer(wf)

        # 读取并写入表头
        header = next(reader)
        writer.writerow(header)

        for row in reader:
            if len(row) < 2:
                writer.writerow(row)
                continue
            idx = row[0]
            summary = row[1]
            comment = process_text_to_comment(summary)
            # 覆盖第二列
            row[1] = comment
            writer.writerow(row)

    # 用临时文件替换原文件
    import os
    os.replace(tmp_file, input_csv)


if __name__ == '__main__':
    # 示例用法
    beautify_csv('../Result/generation/python/SCSL/zero_shot.csv')
