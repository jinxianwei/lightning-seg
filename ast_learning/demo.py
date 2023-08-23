import ast

import astor

# source_file 是任何一个.py文件的路径

with open('/home/bennie/bennie/lightning-seg/ast_learning/source_code.py',
          encoding='utf-8') as f:
    source_code = f.read()
tree = ast.parse(source_code)

import_nodes = []
empty_lines = []
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == 'Classification_2d':
        class_node = node
    elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
        import_nodes.append(node)
    if isinstance(node, ast.Expr) and isinstance(
            node.value, ast.Str) and not node.value.s.strip():
        empty_lines.append(node.lineno)

copied_class_node = ast.copy_location(class_node, ast.ClassDef())
# 替换类节点中的__init__中的内容
for stmt in copied_class_node.body:
    if isinstance(stmt, ast.FunctionDef) and stmt.name == '__init__':
        for sub_stmt in stmt.body:
            # 遍历__init__中的所有操作（super，赋值等）
            if isinstance(sub_stmt, ast.Assign) and len(
                    sub_stmt.targets) == 1 and isinstance(
                        sub_stmt.targets[0],
                        ast.Attribute) and sub_stmt.targets[0].attr == 'net':
                sub_stmt.value = ast.parse(
                    'models.convnext_large(pretrained=False)').body[0].value
                # 下面的方式会更改原来的sub_stmt.value 的 type 从_ast.Call object
                # 变为 _ast.Name object 但 也是能用的
                # sub_stmt.value = ast.Name(id='models.resnet50(
                # pretrained=False)',
                # ctx=ast.Load(models.resnet50))
            if isinstance(sub_stmt, ast.Assign) and len(
                    sub_stmt.targets) == 1 and isinstance(
                        sub_stmt.targets[0],
                        ast.Attribute) and sub_stmt.targets[0].attr == 'loss':
                sub_stmt.value = ast.parse('nn.CrossEntropyLoss').body[0].value
                # ast.parse不会改变node的type，
                # 几种其他方式的mode赋值
                # sub_stmt.value = ast.Name(id='nn.L1Loss', ctx=ast.Load())
                # 会更改原本的value的type从_ast.Attribute object 变为_ast.Name object

code_tree = ast.Module(body=import_nodes + [copied_class_node])
# 四个空格作为每级缩进
copied_code = astor.to_source(code_tree, indent_with=' ' * 4)
with open('/home/bennie/bennie/lightning-seg/ast_learning/target_code.py',
          'w') as f:
    f.write(copied_code)
