import ast
import sys
import os

def analyze_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        try:
            tree = ast.parse(f.read(), filename=filepath)
        except SyntaxError:
            return
    
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = node.lineno
            end = node.end_lineno
            length = end - start + 1
            if length > 40:
                print(f"{filepath}:{start} - {node.name} ({length} lines)")

for root, _, files in os.walk('src'):
    for file in files:
        if file.endswith('.py'):
            analyze_file(os.path.join(root, file))
for root, _, files in os.walk('scripts'):
    for file in files:
        if file.endswith('.py') and not 'archived' in root:
            analyze_file(os.path.join(root, file))
