import re
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data

from src.shared.preprocessing.math_tokenizer import MathTokenizer

# Node
class ExprNode:
    def __init__(self, value: str, children: List['ExprNode'] = None):
        self.value = value  # Giá trị của node (số, biến, toán tử, v.v.)
        self.children = children if children is not None else []

    def __repr__(self):
        return f"ExprNode(value={self.value}, children={self.children})"
    
# Parser: Chuyển LaTeX thành Expression Tree
class ExprTreeParser:
    def __init__(self, tokenizer: MathTokenizer):
        self.tokenizer = tokenizer
        self.operators = {
            '=': 0, '>': 0, '<': 0, '\\geq': 0, '\\leq': 0, 
            '+': 1, '-': 1, '\\times': 2, '\\div': 2, '\\cdot': 2, '^': 3
        }
        self.latex_commands = [
            '\\frac', '\\sqrt', '\\log', '\\sin', '\\cos', '\\tan', '\\exp', '\\ln',
            '\\Delta', '\\wedge', '\\infty', '\\begin', '\\end', '\\\\', 'cases',
            '\\left', '\\right'
        ]
        self.unary_latex = ['\\sqrt', '\\sin', '\\cos', '\\tan', '\\exp', '\\ln', '\\log']
        self.open_brackets = ['(', '{', '[', '\\left']
        self.close_brackets = [')', '}', ']', '\\right']

    def to_expr_tree(self, tokens: List[str], debug: bool = False) -> ExprNode:
        def precedence(op: str) -> int:
            return self.operators.get(op, 0)
        
        def apply_operator(nodes: List[ExprNode], op: str) -> bool:
            if len(nodes) < 2:
                if debug:
                    print(nodes, op)
                print(f"Error: Not enough nodes for operator {op}")
                return False
            right, left = nodes.pop(), nodes.pop()
            nodes.append(ExprNode(op, [left, right]))
            return True
        
        def is_unary_minus(i: int, tokens: List[str]) -> bool:
            if i == 0:
                return True
            prev = tokens[i - 1]
            return prev in self.operators or prev in self.open_brackets
        
        def is_number(value: str) -> bool:
            return value.lstrip('-').isdigit() or value.replace('.', '', 1).lstrip('-').isdigit()

        def build_tree(index: int, end: int, debug: bool = False) -> Tuple[ExprNode, int]:
            nodes, ops = [], []
            i, in_number, number_tokens = index, False, []

            while i < end:
                token = tokens[i]
                # print(f"Processing token {i}: {token}, nodes={nodes}, ops={ops}")
                if token == '<NUM>':
                    in_number, number_tokens = True, []
                    i += 1
                    continue
                elif token == '</NUM>':
                    nodes.append(ExprNode(''.join(number_tokens)))
                    in_number, i = False, i + 1
                    if debug:
                        print(f"DEBUG: Added number node {nodes[-1].value}, nodes={[n.value for n in nodes]}")
                    continue
                elif in_number:
                    number_tokens.append(token)
                    i += 1
                    continue
                elif token in self.open_brackets:
                    sub_tree, i = build_tree(i + 1, end, debug)
                    if sub_tree:
                        nodes.append(sub_tree)
                        if debug:
                            print(f"DEBUG: Added subtree {sub_tree.value}, nodes={[n.value for n in nodes]}")
                    if i < end and tokens[i] in self.close_brackets:
                        i += 1  # Bỏ qua dấu đóng
                        
                    # Thêm phép nhân ngầm nếu trước ngoặc là số hoặc biến
                    if nodes and len(nodes) >= 2:
                        last_node = nodes[-2]  # Node trước ngoặc
                        if (is_number(last_node.value) or last_node.value.lstrip('-') in ['x', 'y', 'z']):
                            ops.append('\\times')
                            if debug:
                                print(f"DEBUG: Added implicit \\times after bracket, ops={ops}")
                    continue
                elif token in self.close_brackets:
                    while ops:
                        if not apply_operator(nodes, ops.pop()):
                            return None, i
                    if debug:
                        print(f"DEBUG: Closed bracket, nodes={[n.value for n in nodes]}")
                    return nodes[0] if nodes else None, i
                elif token in self.latex_commands:
                    if token == '\\frac':
                        i += 1
                        if i >= end or tokens[i] != '{':
                            print("Error: Expected '{' after \\frac")
                            return None, i
                        num_tree, i = build_tree(i + 1, end, debug)
                        if not num_tree or i >= end or tokens[i] != '}':
                            print("Error: Invalid numerator")
                            return None, i
                        i += 1
                        if i >= end or tokens[i] != '{':
                            print("Error: Expected '{' for denominator")
                            return None, i
                        den_tree, i = build_tree(i + 1, end, debug)
                        if not den_tree or i >= end or tokens[i] != '}':
                            print("Error: Invalid denominator")
                            return None, i
                        nodes.append(ExprNode('\\frac', [num_tree, den_tree]))
                        i += 1
                        if debug:
                            print(f"DEBUG: Created \\frac node, nodes={[n.value for n in nodes]}")
                        continue
                    elif token in self.unary_latex:
                        i, children = i + 1, []
                        if i >= end:
                            print(f"Error: Expected argument for {token}")
                            return None, i
                        # Xử lý \log_{base}{argument}
                        if token == '\\log' and i < end and tokens[i] == '_':
                            i += 1
                            if i >= end or tokens[i] != '{':
                                print("Error: Expected '{' for \\log base")
                                return None, i
                            base_tree, i = build_tree(i + 1, end, debug)
                            if not base_tree or i >= end or tokens[i] != '}':
                                print("Error: Invalid \\log base")
                                return None, i
                            i, children = i + 1, [base_tree]
                        # Xử lý đối số chính
                        if i < end and tokens[i] in self.open_brackets:
                            sub_tree, i = build_tree(i + 1, end, debug)
                            if i < end and tokens[i] in self.close_brackets:
                                i += 1
                        else:
                            # Đối số là token đơn hoặc số
                            if i < end and tokens[i] == '<NUM>':
                                num_tokens, i = [], i + 1
                                while i < end and tokens[i] != '</NUM>':
                                    num_tokens.append(tokens[i])
                                    i += 1
                                if i < end:
                                    i += 1
                                sub_tree = ExprNode(''.join(num_tokens))
                            else:
                                sub_tree, i = ExprNode(tokens[i]), i + 1
                        children.append(sub_tree)
                        nodes.append(ExprNode(token, children))
                        if debug:
                            print(f"DEBUG: Added unary latex {token}, nodes={[n.value for n in nodes]}")
                        continue
                    elif token in ['\\begin', '\\end']:
                        i += 1
                        continue
                    else:
                        nodes.append(ExprNode(token))
                        i += 1
                        if debug:
                            print(f"DEBUG: Added latex command {token}, nodes={[n.value for n in nodes]}")
                        continue
                elif token in self.operators:
                    if token == '-':
                        if is_unary_minus(i, tokens):
                            i += 1
                            if i < end:
                                if tokens[i] == '<NUM>':
                                    num_tokens, i = [], i + 1
                                    while i < end and tokens[i] != '</NUM>':
                                        num_tokens.append(tokens[i])
                                        i += 1
                                    if i < end:
                                        i += 1
                                    nodes.append(ExprNode('-' + ''.join(num_tokens)))
                                    if debug:
                                        print(f"DEBUG: Added unary minus number -{nodes[-1].value}, nodes={[n.value for n in nodes]}")
                                elif tokens[i].isalnum() or tokens[i] in ['x', 'y', 'z']:
                                    nodes.append(ExprNode('-' + tokens[i]))
                                    i += 1
                                    if debug:
                                        print(f"DEBUG: Added unary minus variable -{nodes[-1].value}, nodes={[n.value for n in nodes]}")
                                else:
                                    print(f"Error: Invalid operand after unary minus at token {i}")
                                    return None, i
                            else:
                                print("Error: Expected operand after unary minus")
                                return None, i
                        # Binary minus
                        else:
                            while ops and ops[-1] not in ['=', '>', '<', '\\geq', '\\leq'] and \
                                (precedence(ops[-1]) >= precedence(token)):
                                if not apply_operator(nodes, ops.pop()):
                                    return None, i
                            ops.append(token)
                            i += 1
                            if debug:
                                print(f"DEBUG: Added binary operator -, ops={ops}")
                        continue
                    elif token == '=':
                        # Áp dụng tất cả toán tử trước khi thêm '='
                        while ops:
                            if not apply_operator(nodes, ops.pop()):
                                return None, i
                        if not nodes:
                            print("Error: No expression before '='")
                            return None, i
                        right_tree, i = build_tree(i + 1, end, debug)
                        if right_tree:
                            nodes.append(right_tree)
                        else:
                            nodes.append(ExprNode('0'))
                        if len(nodes) == 2:
                            left, right = nodes
                            nodes = [ExprNode('=', [left, right])]
                        else:
                            print(nodes, ops)
                            print("Error: Invalid nodes for '='")
                            return None, i
                        if debug:
                            print(f"DEBUG: Created = node, nodes={[n.value for n in nodes]}")
                        return nodes[0], i
                    else:
                        # Other operators
                        while ops and ops[-1] not in ['=', '>', '<', '\\geq', '\\leq'] and \
                            (precedence(ops[-1]) >= precedence(token)):
                            if not apply_operator(nodes, ops.pop()):
                                return None, i
                        ops.append(token)
                        i += 1
                        if debug:
                            print(f"DEBUG: Added operator {token}, ops={ops}")
                        continue
                else:
                    # Thêm phép nhân ngầm nếu trước là số hoặc biến
                    if nodes and token not in self.operators and token not in self.latex_commands:
                        last_node = nodes[-1]
                        if is_number(last_node.value) or last_node.value.lstrip('-') in ['x', 'y', 'z']:
                            if (token.isalnum() or token in ['x', 'y', 'z'] or token in self.open_brackets) and \
                               (i + 1 >= end or tokens[i - 1] not in self.operators):
                                ops.append('\\times')
                                if debug:
                                    print(f"DEBUG: Added implicit \\times for {last_node.value}{token}, ops={ops}")
                    nodes.append(ExprNode(token))
                    i += 1
                    if debug:
                        print(f"DEBUG: Added node {token}, nodes={[n.value for n in nodes]}")

            while ops:
                op = ops.pop()
                if not apply_operator(nodes, op):
                    return None, i
                
            return nodes[0] if nodes else None, i

        tree, _ = build_tree(0, len(tokens), debug)
        return tree

    def tree_to_graph(self, tree: 'ExprNode', token_to_idx: Dict[str, int]) -> Data:
        nodes = []
        edges = []
        node_features = []
        node_idx = {}
        missing_tokens = set()

        def is_number(value: str) -> bool:
            return bool(re.match(r'^-?\d+$', value))
        
        def is_variable(value: str) -> bool:
            return value.lstrip('-') in ('x', 'y', 'z')

        def tokenize_number(value: str) -> List[str]:
            """Tách số hoặc biến thành dấu và thành phần."""
            tokens = []
            if value.startswith('-'):
                tokens.append('-')
                value = value[1:]
            if is_number(value):
                tokens.extend(list(value))  # Tách số thành chữ số
            else:
                tokens.append(value)  # Giữ biến nguyên vẹn
            return tokens

        def traverse(node: 'ExprNode', parent_idx: int = -1):
            current_idx = len(nodes)
            # print(f"Processing node: {node.value}, parent_idx: {parent_idx}")

            # Xử lý node.value
            if is_number(node.value) or is_variable(node.value):
                # Tách số hoặc biến
                num_tokens = tokenize_number(node.value)
                num_indices = []
                for token in num_tokens:
                    if token not in token_to_idx:
                        missing_tokens.add(token)
                    node_features.append(token_to_idx.get(token, token_to_idx['<unk>']))
                    nodes.append(token)
                    num_indices.append(len(nodes) - 1)
                
                # Thêm cạnh nối các chữ số trong số
                for i in range(1, len(num_indices)):
                    edge = [num_indices[i-1], num_indices[i]]
                    edges.append(edge)
                
                if parent_idx != -1 and num_indices:
                    edge = [parent_idx, num_indices[0]]
                    edges.append(edge)
                
                for child in node.children:
                    traverse(child, num_indices[-1] if num_indices else parent_idx)
            else:
                # Xử lý token bình thường
                if node.value not in node_idx:
                    node_idx[node.value] = len(nodes)
                    nodes.append(node.value)
                    if node.value not in token_to_idx:
                        missing_tokens.add(node.value)
                    node_features.append(token_to_idx.get(node.value, token_to_idx['<unk>']))
                current_idx = node_idx[node.value]
                
                if parent_idx != -1 and current_idx != parent_idx:
                    edge = [parent_idx, current_idx]
                    edges.append(edge)
                
                for child in node.children:
                    traverse(child, current_idx)

        traverse(tree)
        if missing_tokens:
            print(f"Missing tokens in token_to_idx: {missing_tokens}")

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(node_features, dtype=torch.long)
        return Data(x=x, edge_index=edge_index)

    def print_graph(self, graph_data: Data, tokenizer: MathTokenizer):
        print("Nodes (token ID, token value):")
        for i, node_id in enumerate(graph_data.x):
            token = tokenizer.idx_to_token.get(node_id.item(), '<unk>')
            print(f"Node {i}: ID={node_id.item()}, Value={token}")
        print("\nEdges (from, to):")
        for edge in graph_data.edge_index.t():
            print(f"Edge: {edge[0].item()} -> {edge[1].item()}")

    def visualize_graph(self, graph_data: Data, tokenizer: MathTokenizer, title: str = "Expression Tree"):
        G = nx.Graph()
        node_labels = {}
        for i, node_id in enumerate(graph_data.x):
            token = tokenizer.idx_to_token.get(node_id.item(), '<unk>')
            G.add_node(i)
            node_labels[i] = token
        for edge in graph_data.edge_index.t():
            G.add_edge(edge[0].item(), edge[1].item())

        plt.figure(figsize=(8, 6))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', 
                node_size=500, font_size=10, font_weight='bold')
        plt.title(title)
        plt.savefig('graph.png')
        plt.close()