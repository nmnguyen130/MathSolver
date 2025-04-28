import json
from typing import Dict, List
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import re

from src.shared.preprocessing.math_tokenizer import MathTokenizer

class MathDataset(Dataset):
    def __init__(self, json_file: str, tokenizer: MathTokenizer, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = tokenizer

        # Read dataset from JSON file
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Build vocabulary from dataset
        self.tokenizer.build_vocab(self.data)

    def _clean_latex(self, latex_str: str) -> str:
        """Preprocess LaTeX string to make it compatible with SymPy's parse_latex."""
        # Remove <NUM> tags if present
        latex_str = re.sub(r'<NUM>\s*([\d\s]+)\s*</NUM>', r'\1', latex_str)
        # Replace multiple spaces with a single space
        latex_str = re.sub(r'\s+', ' ', latex_str)
        # Handle double negatives (e.g., "- -31" -> "- (-31)")
        latex_str = re.sub(r'-\s*-(\d+)', r'- (-\1)', latex_str)
        # Ensure proper LaTeX syntax for equations
        latex_str = latex_str.strip()
        return latex_str

    def _sympy_to_graph(self, expr, token_to_idx: Dict[str, int]) -> Data:
        """Convert a SymPy expression to a PyTorch Geometric graph."""
        nodes = []
        edge_index = []
        node_indices = {}
        current_idx = 0

        def traverse(expr, parent_idx=None):
            nonlocal current_idx
            # Get string representation of the expression
            expr_str = str(expr)
            if expr.is_Atom:
                node_val = expr_str
            else:
                node_val = expr.func.__name__

            # Add node if not already added
            if expr not in node_indices:
                node_indices[expr] = current_idx
                nodes.append(token_to_idx.get(node_val, token_to_idx.get('<unk>')))
                current_idx += 1

            current_idx_local = node_indices[expr]

            # Add edge from parent to current node
            if parent_idx is not None:
                edge_index.append([parent_idx, current_idx_local])
                edge_index.append([current_idx_local, parent_idx])  # Bidirectional

            # Process arguments/children
            if not expr.is_Atom:
                for arg in expr.args:
                    traverse(arg, current_idx_local)

        # Traverse the expression tree
        traverse(expr)

        # Convert to tensors
        if not nodes:
            return Data(
                x=torch.tensor([], dtype=torch.long),
                edge_index=torch.tensor([[], []], dtype=torch.long)
            )

        x = torch.tensor(nodes, dtype=torch.long)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        return Data(x=x, edge_index=edge_index)

    def _create_combined_graph(self, step_latex_expressions: List[str]) -> Data:
        """Create a combined graph from multiple LaTeX expressions."""
        graphs = []
        root_indices = []
        node_offset = 0

        for step_latex in step_latex_expressions:
            try:
                # Clean the LaTeX expression
                cleaned_latex = self._clean_latex(step_latex)
                # Parse cleaned LaTeX to SymPy expression
                sympy_expr = parse_latex(cleaned_latex)
                
                # Convert SymPy expression to graph
                graph = self._sympy_to_graph(sympy_expr, self.tokenizer.token_to_idx)
                
                if graph.x.size(0) > 0:
                    graph.edge_index = graph.edge_index + node_offset
                    graphs.append(graph)
                    root_indices.append(node_offset)
                    node_offset += graph.x.size(0)
                else:
                    print(f"Warning: Empty graph for {step_latex}")
            except Exception as e:
                print(f"Warning: Failed to parse {step_latex}: {str(e)}")
                # Fallback: Create a minimal graph with a single node
                fallback_node = torch.tensor([self.tokenizer.token_to_idx.get('<unk>')], dtype=torch.long)
                fallback_graph = Data(
                    x=fallback_node,
                    edge_index=torch.tensor([[], []], dtype=torch.long)
                )
                fallback_graph.edge_index = fallback_graph.edge_index + node_offset
                graphs.append(fallback_graph)
                root_indices.append(node_offset)
                node_offset += 1

        if not graphs:
            print("Warning: No valid graphs created")
            return Data(
                x=torch.tensor([], dtype=torch.long),
                edge_index=torch.tensor([[], []], dtype=torch.long)
            )

        # Combine node features and edges
        x = torch.cat([g.x for g in graphs], dim=0)
        edge_index = torch.cat([g.edge_index for g in graphs], dim=1)

        # Add edges between consecutive root nodes
        extra_edges = []
        for i in range(1, len(graphs)):
            prev_root_idx = root_indices[i-1]
            curr_root_idx = root_indices[i]
            extra_edges.append([prev_root_idx, curr_root_idx])
            extra_edges.append([curr_root_idx, prev_root_idx])  # Bidirectional
        
        if extra_edges:
            extra_edges = torch.tensor(extra_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, extra_edges], dim=1)

        return Data(x=x, edge_index=edge_index)

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