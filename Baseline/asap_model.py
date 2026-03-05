import sys
import tree_sitter_c
import tree_sitter_java
import tree_sitter_python
from tree_sitter import Language, Parser

from util.remove_comments import tree_to_token_index, index_to_code_token, remove_comments_and_docstrings

PY_LANGUAGE = Language(tree_sitter_python.language())
JAVA_LANGUAGE = Language(tree_sitter_java.language())
C_LANGUAGE = Language(tree_sitter_c.language())


def repo_info(data):
    repo = {
        'Repo': data['repo'],
        'Path': data['path'],
        'Function_name': data['func']
    }
    str_repo = '\n'.join([f'{k}: {v}' for k, v in repo.items()])
    return str_repo


def code_modeling(language, data):
    code = data["code"]
    if language == 'java':
        parser = Parser(JAVA_LANGUAGE)
    elif language == 'python':
        parser = Parser(PY_LANGUAGE)
    elif language == 'c':
        parser = Parser(C_LANGUAGE)
    else:
        sys.exit('Language not recognized')
    tree = parser.parse(bytes(code, "utf-8"))
    root_node = tree.root_node
    return root_node


def code_tokens(root_node, data):
    code = data['code']
    tokens_index = tree_to_token_index(root_node)
    code = code.split('\n')
    tokens = [index_to_code_token(x, code) for x in tokens_index]
    return tokens


def dataflow(language, root_node, data):
    code = data["code"]
    dataflow_query = ""

    if language == JAVA_LANGUAGE:
        dataflow_query = """
        [
          (type_identifier) @type_name
          (formal_parameter (identifier) @var_name)
          (local_variable_declaration (variable_declarator (identifier) @var_name))
        ]
        """
    elif language == PY_LANGUAGE:
        dataflow_query = """
        [
          (function_definition name: (identifier) @func_name)
          (parameters (identifier) @var_name)
          (parameters (default_parameter name: (identifier) @var_name))
          (assignment left: (identifier) @var_name)
          (for_statement (identifier) @var_name)
          (keyword_argument value: (identifier) @var_name)
        ]
        """

    dataflow_captures = language.query(dataflow_query).captures(root_node)

    type_name = []
    var_name = []

    for name, node_list in dataflow_captures.items():
        for node in node_list:
            text = code[node.start_byte:node.end_byte]
            if name == 'type_name':
                type_name.append(text)
            elif name == 'var_name':
                var_name.append(text)

    tokens = code_tokens(root_node, data)
    df = {}

    for ty in type_name:
        flow = [i for i, val in enumerate(tokens) if val == ty]
        if len(flow) > 1:
            df[ty] = flow

    for var in var_name:
        flow = [i for i, val in enumerate(tokens) if val == var]
        if len(flow) > 1:
            df[var] = flow

    # 格式化输出
    formatted = []
    for k, v in df.items():
        formatted.append(f"{k}[{v[0]}]{v[1:]}")

    return '\n'.join(formatted)


def scope(language, root_node, data):
    code = data["code"]
    func = data["func"]
    # 定义目标查询模式
    scopes_query = ""
    if language == JAVA_LANGUAGE:
        scopes_query = """
        [
          ; 参数名
          (method_declaration
            parameters: (formal_parameters
                          (formal_parameter
                            name: (identifier) @param_name)))
        
          (return_statement (_) @return_id)
        
          ; 方法调用（方法名）
          (method_invocation
            name: (identifier) @method_name)
        
          ; 方法调用的参数
          (method_invocation
            arguments: (argument_list
                         (expression) @method_arg))
        
          ; 局部变量声明
          (local_variable_declaration
            (variable_declarator
              name: (identifier) @var_del))
        ]
        """

        # 提取变量声明
        scopes_captures = language.query(scopes_query).captures(root_node)

        param_name, return_id, method_name, method_args, var_del = [], [], [], [], []

        for name, node_list in scopes_captures.items():
            for node in node_list:
                text = code[node.start_byte:node.end_byte]
                if name == 'param_name':
                    param_name.append(text)
                elif name == 'return_id':
                    return_id.append(text)
                elif name == 'method_name':
                    method_name.append(text)
                elif name == 'method_arg':
                    method_args.append(text)
                elif name == 'var_del':
                    var_del.append(text)
        param_name = list(set(param_name))
        return_id = list(set(return_id))
        method_name = list(set(method_name))
        method_args = list(set(method_args))
        var_del = list(set(var_del))

        scopes = {'Function name': func, 'Parameters of the function': param_name,
                  'Identifier to be returned': return_id,
                  'Method Invocation': method_name, 'Method Arguments': method_args, 'Variable Declaration': var_del}
        scopes = '\n'.join([f'{k}: {v}' for k, v in scopes.items()])
        # print(scopes)
        return scopes

    elif language == PY_LANGUAGE:
        scopes_query = """
        (function_definition
          parameters: (parameters
                       (_) @param))
        
        (return_statement
          (identifier) @return_id)
        
        (call
          function: (identifier) @func_call)
        (call
          function: (attribute
                      object: (identifier) @call_obj
                      attribute: (identifier) @call_method))
        
        (assignment
          left: (identifier) @assigned_var)
        
        (attribute
          object: (identifier) @attr_obj
          attribute: (identifier) @attr_name)
        """

    # 提取变量声明
        scopes_captures = language.query(scopes_query).captures(root_node)

        param_name, return_id, method_name, method_args, var_del = [], [], [], [], []

        for name, node_list in scopes_captures.items():
            for node in node_list:
                text = code[node.start_byte:node.end_byte]
                if name == 'param':
                    param_name.append(text)
                elif name == 'return_id':
                    return_id.append(text)
                elif name == 'func_call':
                    method_name.append(text)
                elif name == 'call_obj' or name == 'call_method':
                    method_name.append(text)
                elif name == 'assigned_var':
                    method_args.append(text)
                elif name == 'attr_obj' or name == 'attr_name':
                    var_del.append(text)

        param_name = list(set(param_name))
        return_id = list(set(return_id))
        method_name = list(set(method_name))
        method_args = list(set(method_args))
        var_del = list(set(var_del))


        scopes = {'Function name': func, 'Parameters of the function': param_name, 'Identifier to be returned': return_id,
                  'Method Invocation': method_name, 'Assignments': method_args, 'Identifier to access attribute/dotted name': var_del}
        scopes = '\n'.join([f'{k}: {v}' for k, v in scopes.items()])
        # print(scopes)
        return scopes


def asap(lang, data, type=None):
    language = ""
    if lang == 'python':
        language = PY_LANGUAGE
    elif lang == 'java':
        language = JAVA_LANGUAGE
    if type == 'dataflow':
        df = dataflow(language, code_modeling(lang, data), data)
        return df
    ri = repo_info(data)
    df = dataflow(language, code_modeling(lang, data), data)
    sp = scope(language, code_modeling(lang, data), data)
    result = {'Code': data['code'], 'Repo_info': ri, 'Dataflow': df, 'Scopes': sp, 'Output': data['comment']}
    return result


