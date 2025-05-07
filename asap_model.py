import json
import sys
from nltk.text import Text
import nltk
import tree_sitter_c
import tree_sitter_java
import tree_sitter_python
from tree_sitter import Language, Parser
import tiktoken
from nltk.tokenize import word_tokenize

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


def walk(node, tokens):
    if node.child_count == 0:  # 叶子节点即标记
        s = node.text.decode("utf8")
        tokens.append(s)
    else:
        for child in node.children:
            walk(child, tokens)


def dataflow(language, root_node, data):
    code = data["code"]
    # 查询变量声明（参数 + 局部变量）
    dataflow_query = ""
    if language == JAVA_LANGUAGE:
        dataflow_query = """
        [
          (type_identifier) @type_name; 类型声明和使用
          (formal_parameter (identifier) @var_name) @param_node; 方法参数
          (local_variable_declaration (variable_declarator (identifier) @var_name)) @local_var_node ; 局部变量
        ]
        """
    elif language == PY_LANGUAGE:
        dataflow_query = """
                [
                    (function_definition name: (identifier) @func_name)
                    (parameters (identifier) @var_name) ; 方法参数
                    (assignment left: (identifier) @var_name) ; 变量赋值
                    (type (identifier) @type_name) ; 类型注解中的类型
                ]
                """
    # 提取变量声明
    dataflow_captures = language.query(dataflow_query).captures(root_node)

    type_name = []
    var_name = []
    # param_node = []
    # local_var_node = []
    for node, name in dataflow_captures:
        if name == 'type_name':
            type_name.append(code[node.start_byte:node.end_byte])
            # print(node.start_point)
        elif name == 'var_name':
            var_name.append(code[node.start_byte:node.end_byte])
            # print(node.start_point)
        # elif name == 'param_node':
        #     param_node.append(code[node.start_byte:node.end_byte])
        # elif name == 'local_var_node':
        #     local_var_node.append(code[node.start_byte:node.end_byte])

    # print(type_name)
    # print(var_name)
    # print(param_node)
    # print(local_var_node)

    tokens = []
    walk(root_node, tokens)

    # print(tokens)
    df = {}
    for ty in type_name:
        flow = []
        for index, value in enumerate(tokens):
            if value == ty:
                flow.append(index)
        df[ty] = flow
    for var in var_name:
        flow = []
        for index, value in enumerate(tokens):
            if value == var:
                flow.append(index)
        df[var] = flow
    df = '\n'.join([f'{k}: {v}' for k, v in df.items()])
    # print(df)
    return df


def scope(language, root_node, data):
    code = data["code"]
    func = data["func"]
    # 定义目标查询模式
    scopes_query = ""
    if language == JAVA_LANGUAGE:
        scopes_query = """
        [
          (method_declaration (formal_parameters (formal_parameter (identifier) @param_name)))
          (return_statement (_) @return_id)
          (method_invocation (_) @method_name)
          (method_invocation (identifier) (argument_list) @method_args)
          (local_variable_declaration (variable_declarator (identifier) @var_del))
        ]
        """
    elif language == PY_LANGUAGE:
        scopes_query = """
        [
            (function_definition parameters: (parameters (identifier) @param_name))
            (return_statement (_) @return_id)
            (call function: (_) @method_name)
            (call arguments: (argument_list) @method_args)
            (assignment left: (identifier) @var_del)
        ]
        """
    # 提取变量声明
    scopes_captures = language.query(scopes_query).captures(root_node)

    param_name, return_id, method_name, method_args, var_del = [], [], [], [], []

    for node, name in scopes_captures:
        if name == 'param_name':
            param_name.append(code[node.start_byte:node.end_byte])
        elif name == 'return_id':
            return_id.append(code[node.start_byte:node.end_byte])
        elif name == 'method_name':
            method_name.append(code[node.start_byte:node.end_byte])
        elif name == 'method_args':
            method_args.append(code[node.start_byte:node.end_byte])
        elif name == 'var_del':
            var_del.append(code[node.start_byte:node.end_byte])

    scopes = {'Function name': func, 'Parameters of the function': param_name, 'Identifier to be returned': return_id,
              'Method Invocation': method_name, 'Method Arguments': method_args, 'Variable Declaration': var_del}
    scopes = '\n'.join([f'{k}: {v}' for k, v in scopes.items()])
    # print(scopes)
    return scopes


def asap(lang, data):
    language = ""
    if lang == 'python':
        language = PY_LANGUAGE
    elif lang == 'java':
        language = JAVA_LANGUAGE
    ri = repo_info(data)
    df = dataflow(language, code_modeling(lang, data), data)
    sp = scope(language, code_modeling(lang, data), data)
    result = {'Code': data['code'], 'Repo_info': ri, 'Dataflow': df, 'Scopes': sp, 'Output': data['comment']}
    return result


# ========================== 使用示例 ==========================
if __name__ == "__main__":
    data = {"code": '''
    def _element_to_dict(data, position, obj_end, opts):
        element_type = data[position:position + 1]
        position += 1
        element_name, position = _get_c_string(data, position, opts)
        try:
            value, position = _ELEMENT_GETTER[element_type](data, position,
                                                            obj_end, opts,
                                                            element_name)
        except KeyError:
            _raise_unknown_type(element_type, element_name)
        
         return element_name, value, position
        
''',"func":"Util.deserializeOffsetMap","repo":"1","path":"2","comment":"3"}
    data1 = {"code": """
    @SuppressWarnings("unchecked")
    public static Map < String,String > deserializeOffsetMap(String lastSourceOffset) throws IOException{
        Map < String,String > offsetMap;
        if(lastSourceOffset == null || lastSourceOffset.isEmpty()){
            offsetMap = new HashMap<>();
        }else{
            offsetMap = JSON_MAPPER.readValue(lastSourceOffset,Map.class);
        }
        return 0;
    }
    """, "func": "Util.deserializeOffsetMap"}

    print(asap('python', data))
    # print(asap('java', data1))
    # encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # tokens = encoding.encode(data["code"])
    # i = 0
    # for token in tokens:
    #     i+=1
    #     print(f"{i}:  {encoding.decode_single_token_bytes(token)}")
    # 首次运行需要下载punkt分词模型
