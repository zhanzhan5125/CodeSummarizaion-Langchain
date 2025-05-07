import re
from io import StringIO
import tokenize

single = {'java': '//', 'python': '#', 'c': '//'}

multi = {'java': ['/*', '*/'], 'python': ['"""', '"""', '/*', '*/'], 'c': ['/*', '*/']}


def remove_comments_and_docstrings(source, lang):
    if lang in ['python']:
        """
        Returns "source" minus comments and docstrings.
        """
        io_obj = StringIO(source)
        out = ""
        prev_toktype = tokenize.INDENT
        last_lineno = -1
        last_col = 0
        for tok in tokenize.generate_tokens(io_obj.readline):
            token_type = tok[0]
            token_string = tok[1]
            start_line, start_col = tok[2]
            end_line, end_col = tok[3]
            ltext = tok[4]
            if start_line > last_lineno:
                last_col = 0
            if start_col > last_col:
                out += (" " * (start_col - last_col))
            # Remove comments:
            if token_type == tokenize.COMMENT:
                pass
            # This series of conditionals removes docstrings:
            elif token_type == tokenize.STRING:
                if prev_toktype != tokenize.INDENT:
                    # This is likely a docstring; double-check we're not inside an operator:
                    if prev_toktype != tokenize.NEWLINE:
                        if start_col > 0:
                            out += token_string
            else:
                out += token_string
            prev_toktype = token_type
            last_col = end_col
            last_lineno = end_line
        temp = []
        for x in out.split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)
    elif lang in ['java','c']:
        def replacer(match):
            s = match.group(0)
            if s.startswith('//') or s.startswith('/*'):
                return " "  # note: a space and not an empty string
            else:
                return s
        pattern = re.compile(
            r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
            re.DOTALL | re.MULTILINE
        )
        temp = []
        for x in re.sub(pattern, replacer, source).split('\n'):
            if x.strip() != "":
                temp.append(x)
        return '\n'.join(temp)

    else:
        print('Unkown language!')
        return source


if __name__ == '__main__':
    code = ("def _get_command_by_name(self, blueprint, name):\n        \"\"\"\n        Get the primary key command it it exists.\n        \"\"\"\n        commands = self._get_commands_by_name(blueprint, name)\n\n        if len(commands):\n            return commands[0]")
    print(code)
    print(remove_comments_and_docstrings(code, 'python'))
    code2=("def valid(self, name):\n    \"\"\"Ensure a variable name is valid.\n\n    Note: Assumes variable names are ASCII, which isn't necessarily true in\n    Python 3.\n\n    Args:\n      name: A proposed variable name.\n\n    Returns:\n      A valid version of the name.\n    \"\"\"\n    name = re.sub('[^0-9a-zA-Z_]', '', name)\n    if re.match('[0-9]', name):\n      name = '_' + name\n    return name")
    print(code2)
    print(remove_comments_and_docstrings(code2, 'python'))