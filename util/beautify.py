import csv

def beautify(FILE='', PROMPT='chain_of_thought'):
    files_1 = [
        'cot',
        'critique',
        'expert'
    ]
    if PROMPT in files_1:
        updated_rows = []
        with open(FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                idx = line[0]
                paragraph = line[1].strip().split('\n')
                paragraph = [p for p in paragraph if len(p.strip()) > 0 and not p.strip().endswith(':')]
                comment = ''
                if len(paragraph) > 0:
                    p = paragraph[0].strip()
                    if p.startswith('/*'):
                        if len(p) <= 3:
                            final_lines = []
                            for k in range(len(paragraph)):
                                if paragraph[k].strip().startswith('*/'):
                                    break
                                elif paragraph[k].strip().startswith('*'):
                                    final_lines.append(paragraph[k].strip()[2:])
                                elif paragraph[k].strip().startswith('/*'):
                                    final_lines.append(paragraph[k].strip()[2:])
                                else:
                                    final_lines.append(paragraph[k].strip())
                            sentence = ' '.join(final_lines).split('. ')
                            sentence = [c for c in sentence if len(c.strip()) > 1]
                            comment = sentence[0] if sentence else ''
                        else:
                            comment = p[2:].strip()
                            if comment.endswith('*/'):
                                comment = comment[:-2]
                    elif p.startswith('```'):
                        comment = paragraph[1] if len(paragraph) > 1 else ''
                        flg = False
                        for i in range(1, len(paragraph)):
                            p = paragraph[i].strip()
                            if not flg and (p.startswith('//') or p.startswith('# ')):
                                comment = p
                                break
                            if not flg and p.startswith('"""'):
                                if len(p) == 3:
                                    final_lines = []
                                    for j in range(i + 1, len(paragraph)):
                                        if paragraph[j].strip().startswith('"""'):
                                            break
                                        final_lines.append(paragraph[j].strip())
                                    sentence = ' '.join(final_lines).split('. ')
                                    sentence = [c for c in sentence if len(c.strip()) > 1]
                                    comment = sentence[0] if sentence else ''
                                    break
                                else:
                                    comment = p[3:]
                                    if comment.endswith('"""'):
                                        comment = comment[:-3]
                                    break
                            if not flg and p.startswith('/*'):
                                if len(p) <= 3:
                                    final_lines = []
                                    for j in range(i + 1, len(paragraph)):
                                        if paragraph[j].strip().startswith('*/'):
                                            break
                                        elif paragraph[j].strip().startswith('*'):
                                            final_lines.append(paragraph[j].strip()[2:])
                                        elif paragraph[j].strip().startswith('/*'):
                                            final_lines.append(paragraph[j].strip()[2:])
                                        else:
                                            final_lines.append(paragraph[j].strip())
                                    sentence = ' '.join(final_lines).split('. ')
                                    sentence = [c for c in sentence if len(c.strip()) > 1]
                                    comment = sentence[0] if sentence else ''
                                    break
                                else:
                                    comment = p[2:].strip()
                                    if comment.endswith('*/'):
                                        comment = comment[:-2]
                                    break
                            if flg and len(p.strip()) > 0:
                                sentence = p.strip().split('. ')
                                sentence = [c for c in sentence if len(c.strip()) > 1]
                                comment = sentence[0] if sentence else ''
                                break
                            if p.startswith('```'):
                                flg = True
                    elif p.startswith('"""'):
                        if len(p) == 3:
                            final_lines = []
                            for j in range(1, len(paragraph)):
                                if paragraph[j].strip().startswith('"""'):
                                    break
                                final_lines.append(paragraph[j].strip())
                            sentence = ' '.join(final_lines).split('. ')
                            sentence = [c for c in sentence if len(c.strip()) > 1]
                            comment = sentence[0] if sentence else ''
                        else:
                            comment = p[3:]
                            if comment.endswith('"""'):
                                comment = comment[:-3]
                    else:
                        sentence = p.split('. ')
                        sentence = [c for c in sentence if len(c.strip()) > 1]
                        if PROMPT == 'critique':
                            sentence = [c for c in sentence if 'apologies' not in c.lower()]
                        comment = sentence[0] if sentence else ''

                    comment = comment.strip()
                    if comment.startswith(('"', '`', '-', '#', '*')):
                        comment = comment[1:].strip()
                    if comment.endswith('"'):
                        comment = comment[:-1].strip()
                    if comment.startswith('//') or comment.startswith('""') or comment.startswith('/*'):
                        comment = comment[2:].strip()
                    if comment.endswith('*/'):
                        comment = comment[:-2].strip()
                    if comment.endswith('.'):
                        comment = comment[:-1].strip()
                    comment = comment.replace('\t', ' ')

                updated_rows.append([idx, comment])

        # 覆盖写回CSV
        with open(FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)

    files_2 = ['zero', 'few']
    if PROMPT in files_2:
        updated_rows = []
        with open(FILE, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for line in reader:
                idx = line[0]
                paragraph = line[1].strip().split('\n')
                paragraph = [p for p in paragraph if len(p.strip()) > 0]
                comment = ''
                if paragraph:
                    p = paragraph[0].strip()
                    if p.startswith('/*'):
                        if len(p) <= 3:
                            final_lines = []
                            for k in range(len(paragraph)):
                                if paragraph[k].strip().startswith('*/'):
                                    break
                                elif paragraph[k].strip().startswith('*'):
                                    final_lines.append(paragraph[k].strip()[2:])
                                elif paragraph[k].strip().startswith('/*'):
                                    final_lines.append(paragraph[k].strip()[2:])
                                else:
                                    final_lines.append(paragraph[k].strip())
                            sentence = ' '.join(final_lines).split('. ')
                            sentence = [c for c in sentence if len(c.strip()) > 1]
                            comment = sentence[0] if sentence else ''
                        else:
                            comment = p[2:].strip()
                            if comment.endswith('*/'):
                                comment = comment[:-2]
                    else:
                        sentence = p.split('. ')
                        sentence = [c for c in sentence if len(c.strip()) > 1]
                        comment = sentence[0] if sentence else ''
                        comment = comment.strip()
                        if comment.startswith(('"', '`', '-', '#', '*')):
                            comment = comment[1:].strip()
                        if comment.endswith('"'):
                            comment = comment[:-1].strip()
                        if comment.startswith('//') or comment.startswith('""') or comment.startswith('/*'):
                            comment = comment[2:].strip()
                        if comment.endswith('*/'):
                            comment = comment[:-2].strip()
                        if comment.endswith('.'):
                            comment = comment[:-1].strip()
                        comment = comment.replace('\t', ' ')

                updated_rows.append([idx, comment])

        # 覆盖写回CSV
        with open(FILE, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(updated_rows)



if __name__ == '__main__':
    # 示例用法
    beautify('../Result/RQ1/java/SCSL/gpt-4-turbo/critique/valid.csv', 'critique')
    beautify('../Result/RQ1/java/SCSL/gpt-4-turbo/expert/valid.csv', 'expert')

