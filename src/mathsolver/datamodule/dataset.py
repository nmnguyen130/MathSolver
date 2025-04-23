import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from src.shared.preprocessing.math_tokenizer import MathTokenizer
from src.mathsolver.models.expression_tree import ExprTreeParser

class MathDataset(Dataset):
    def __init__(self, json_file: str, tokenizer: MathTokenizer, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.parser = ExprTreeParser(tokenizer)

        # Đọc dataset từ file JSON
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Xây dựng vocab từ dataset
        self.tokenizer.build_vocab(self.data)

    def _create_combined_graph(self, step_latex_expressions: List[str]) -> Data:
        step_trees = []
        for step_latex in step_latex_expressions:
            step_tokens = self.tokenizer.tokenize_latex(step_latex)
            step_tree = self.parser.to_expr_tree(step_tokens)
            if step_tree is not None:
                step_trees.append(step_tree)
                # print(f"Step tree for {step_latex}:")
                # self._print_tree(step_tree)
            else:
                print(f"Warning: Step tree is None for {step_tokens}")

        combined_nodes = step_trees
        graphs = []
        node_offset = 0
        root_indices = []
        # Tạo graph cho từng cây
        for i, tree in enumerate(combined_nodes):
            graph = self.parser.tree_to_graph(tree, self.tokenizer.token_to_idx)
            graph.edge_index = graph.edge_index + node_offset
            graphs.append(graph)
            root_indices.append(node_offset)
            node_offset += graph.x.size(0)
            # print(f"Graph {i}: {graph.x.size(0)} nodes, root index: {root_indices[-1]}")

        if not graphs:
            print("Warning: No valid graphs created")
            return Data(x=torch.tensor([], dtype=torch.long), edge_index=torch.tensor([[], []], dtype=torch.long))

        # Kết noi các nodes và cạnh
        x = torch.cat([g.x for g in graphs], dim=0)
        edge_index = torch.cat([g.edge_index for g in graphs], dim=1)

        # Thêm cạnh giữa các node gốc trong các cây liên tiếp
        extra_edges = []
        for i in range(1, len(graphs)):
            prev_root_idx = root_indices[i-1]
            curr_root_idx = root_indices[i]
            extra_edges.append([prev_root_idx, curr_root_idx])
        
        if extra_edges:
            extra_edges = torch.tensor(extra_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, extra_edges], dim=1)

        return Data(x=x, edge_index=edge_index)
    
    def _print_tree(self, node, level=0):
        if node is None:
            print("  " * level + "None")
            return
        print("  " * level + node.value)
        for child in node.children:
            self._print_tree(child, level + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.data[idx]
        latex_equation = sample.get("latex_equation", "")
        query = sample.get("query", "")
        solution_steps = sample.get("solution_steps", [])

        input_ids, target_ids, graph_data = self.tokenizer.encode(
            latex_equation=latex_equation,
            query=query,
            solution_steps=solution_steps,
            graph_fn=self._create_combined_graph
        )

        input_ids = input_ids[:self.max_length]
        target_ids = target_ids[:self.max_length]

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        target_ids = torch.tensor(target_ids, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
            'graph_data': graph_data
        }