import re
import logging
from pylatexenc.latexwalker import (LatexWalker, LatexCharsNode, LatexMacroNode, 
                LatexGroupNode, LatexEnvironmentNode, LatexMathNode, LatexSpecialsNode)

def is_ascii(token):
    """Check if all characters in the token are ASCII."""
    return all(ord(c) < 128 for c in token)

class LatexProcessor:
    def __init__(self):
        logging.basicConfig(level=logging.ERROR)
        self.walker = None
        self.norm_str = ""
        # Define node handlers similar to KaTeX's groupTypes
        self.node_handlers = {
            'chars': self.handle_chars,
            'macro': self.handle_macro,
            'group': self.handle_group,
            'environment': self.handle_environment,
            'math': self.handle_math,
            'specials': self.handle_specials,
            'spacing': self.handle_spacing
        }

    def preprocess_latex(self, latex):
        """Step 1: Preprocess LaTeX input to clean and standardize it."""
        # Remove leading comment
        if latex.startswith('%'):
            latex = latex[1:]
        # Remove comments
        latex = latex.split('%')[0].strip()
        # Replace specific commands and symbols
        latex = latex.replace('\\~', ' ')
        for _ in range(300):
            latex = latex.replace('\\>', ' ')
            latex = latex.replace('$', ' ')
            latex = re.sub(r'\\label\{.*?\}', '', latex)
        # Replace \\ with \, if no complex environments
        if not any(x in latex for x in ['matrix', 'cases', 'array', 'begin']):
            for _ in range(300):
                latex = latex.replace('\\\\', '\\,')
        # Replace font commands
        for _ in range(300):
            latex = re.sub(r'{\\rm', r'\\mathrm{', latex)
            latex = re.sub(r'{ \\rm', r'\\mathrm{', latex)
            latex = re.sub(r'\\rm\{', r'\\mathrm{', latex)
        # Replace other commands
        latex = re.sub(r'hskip(.*?)(cm|in|pt|mm|em)', r'\\hspace{\1\2}', latex)
        latex = latex.replace(r'\over', r'\\frac')
        latex = latex.replace(r'\sp', r'^')
        return latex + ' '

    def postprocess_output(self, norm_str):
        """Step 4: Post-process the tokenized output."""
        for _ in range(300):
            norm_str = norm_str.replace('SSSSSS', '$').replace(' S S S S S S', '$')
        norm_str = norm_str.replace(r'\label { .*? }', '')
        return norm_str

    # Node handlers (analogous to KaTeX's groupTypes)
    def handle_chars(self, node, options):
        """Handle character nodes (similar to KaTeX's mathord/textord)."""
        if options.get('font') == 'mathrm':
            # Split characters in mathrm context
            for char in node.chars:
                self.norm_str += char + ' '
                if char == ' ':
                    self.norm_str += '\\; '
        else:
            # Split characters if they contain sub/sup indicators
            chars = node.chars
            i = 0
            while i < len(chars):
                if i + 1 < len(chars) and chars[i] in ['_', '^']:
                    self.norm_str += chars[i] + ' '
                    i += 1
                    # Check if the next part is a group
                    if i < len(chars) and chars[i] == '{':
                        group = ''
                        i += 1
                        depth = 1
                        while i < len(chars) and depth > 0:
                            if chars[i] == '{':
                                depth += 1
                            elif chars[i] == '}':
                                depth -= 1
                            if depth > 0:
                                group += chars[i]
                            i += 1
                        for c in group:
                            self.norm_str += c + ' '
                    elif i < len(chars) and chars[i] != ' ':
                        self.norm_str += '{ ' + chars[i] + ' } '
                        i += 1
                else:
                    self.norm_str += chars[i] + ' '
                    i += 1

    def handle_macro(self, node, options):
        """Handle macro nodes (covers _, ^, frac, sqrt, etc.)."""
        macro_name = node.macroname
        if macro_name in [' ', ',', ';', '!', 'quad', 'qquad', 'enspace', 'thinspace', 'negthinspace']:
            self.handle_spacing(node, options)
        elif macro_name in ['_', '^']:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                arg = node.nodeargs[0]
                needs_braces = isinstance(arg, LatexGroupNode) and len(arg.nodelist) > 1
                if needs_braces:
                    self.norm_str += '{ '
                self.build_group(arg, options)
                if needs_braces:
                    self.norm_str += '} '
        elif macro_name == 'color':
            self.norm_str += '\\color { '
            if node.nodeargs and len(node.nodeargs) > 0:
                self.build_group(node.nodeargs[0], options)  # Color argument
            self.norm_str += '} '
            if len(node.nodeargs) > 1:
                self.build_group(node.nodeargs[1], options)  # Content
        elif macro_name in ['big', 'Big', 'bigg', 'Bigg']:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name in ['displaystyle', 'textstyle', 'scriptstyle', 'scriptscriptstyle']:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name in ['tiny', 'scriptsize', 'footnotesize', 'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge']:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name == 'rule':
            self.norm_str += '\\rule { '
            if node.nodeargs and len(node.nodeargs) > 1:
                self.build_group(node.nodeargs[0], options)  # Width
                self.norm_str += '} { '
                self.build_group(node.nodeargs[1], options)  # Height
                self.norm_str += '} '
        elif macro_name in ['llap', 'rlap']:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name == 'phantom':
            self.norm_str += '\\phantom { '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
            self.norm_str += '} '
        elif macro_name == 'frac':  # Similar to KaTeX's genfrac
            self.norm_str += '\\frac '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)  # Numerator
                self.build_group(node.nodeargs[1], options)  # Denominator
        elif macro_name == 'sqrt':  # Similar to KaTeX's sqrt
            if node.nodeoptarg:
                self.norm_str += '\\sqrt [ '
                self.build_group(node.nodeoptarg, options)
                self.norm_str += '] '
            else:
                self.norm_str += '\\sqrt '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name == 'mathrm':  # Similar to KaTeX's font/text
            self.norm_str += '\\mathrm { '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], {**options, 'font': 'mathrm'})
            self.norm_str += '} '
        elif macro_name in ['bar', 'hat', 'tilde']:  # Similar to KaTeX's accent
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs and len(node.nodeargs) > 0:
                arg = node.nodeargs[0]
                # Only add braces for complex arguments
                needs_braces = isinstance(arg, LatexGroupNode) and len(arg.nodelist) > 1
                if needs_braces:
                    self.norm_str += '{ '
                self.build_group(arg, options)
                if needs_braces:
                    self.norm_str += '} '
        elif macro_name == 'binom':  # Similar to KaTeX's genfrac
            self.norm_str += '\\binom '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
                self.build_group(node.nodeargs[1], options)
        elif macro_name == 'left':  # Similar to KaTeX's leftright
            self.norm_str += '\\left '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name == 'right':
            self.norm_str += '\\right '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
        elif macro_name == 'operatorname':  # Similar to KaTeX's op
            self.norm_str += '\\operatorname { '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
            self.norm_str += '} '
        elif macro_name in ['overline', 'underline']:  # Similar to KaTeX's overline/underline
            self.norm_str += '\\' + macro_name + ' { '
            if node.nodeargs:
                self.build_group(node.nodeargs[0], options)
            self.norm_str += '} '
        else:
            self.norm_str += '\\' + macro_name + ' '
            if node.nodeargs:
                for arg in node.nodeargs:
                    self.build_group(arg, options)

    def handle_group(self, node, options):
        """Handle group nodes (similar to KaTeX's ordgroup)."""
        self.norm_str += '{ '
        for n in node.nodelist:
            self.build_group(n, options)
        self.norm_str += '} '

    def handle_environment(self, node, options):
        """Handle environment nodes (similar to KaTeX's array)."""
        self.norm_str += '\\begin{' + node.environmentname + '} '
        # print(">> ARGNLIST:", node.nodeargd.argnlist)
        if node.nodeargd:
            for arg in node.nodeargd.argnlist:
                if arg is None:
                    continue
                needs_braces = not isinstance(arg, LatexGroupNode)
                if needs_braces:
                    self.norm_str += '{ '
                self.build_group(arg, options)
                if needs_braces:
                    self.norm_str += '} '
        for n in node.nodelist:
            self.build_group(n, options)
        self.norm_str += '\\end{' + node.environmentname + '} '

    def handle_math(self, node, options):
        """Handle math mode nodes."""
        for n in node.nodelist:
            self.build_group(n, options)

    def handle_specials(self, node, options):
        """Handle special character nodes (similar to KaTeX's open/close/punct)."""
        self.norm_str += node.specials_chars + ' '

    def handle_spacing(self, node, options):
        """Handle spacing commands (similar to KaTeX's spacing)."""
        macro_name = node.macroname
        spacing_map = {
            ' ': '\\ ',
            ',': '\\,',
            ';': '\\;',
            '!': '\\!',
            'quad': '\\quad',
            'qquad': '\\qquad',
            'enspace': '\\enspace',
            'thinspace': '\\thinspace',
            'negthinspace': '\\negthinspace',
        }
        if macro_name in spacing_map:
            self.norm_str += spacing_map[macro_name] + ' '
        else:
            self.norm_str += '\\' + macro_name + ' '

    def build_group(self, node, options):
        """Process a single node (similar to KaTeX's buildGroup)."""
        node_type = {
            LatexCharsNode: 'chars',
            LatexMacroNode: 'macro',
            LatexGroupNode: 'group',
            LatexEnvironmentNode: 'environment',
            LatexMathNode: 'math',
            LatexSpecialsNode: 'specials',
        }.get(type(node), 'unknown')
        if node_type in self.node_handlers:
            self.node_handlers[node_type](node, options)
        else:
            logging.warning(f"Unknown node type: {node_type}")
            if hasattr(node, 'nodelist'):
                for n in node.nodelist:
                    self.build_group(n, options)

    def build_expression(self, nodes, options):
        """Process a list of nodes (similar to KaTeX's buildExpression)."""
        for node in nodes:
            self.build_group(node, options)

    def normalize_latex(self, latex):
        """Step 2-4: Parse, render, and post-process LaTeX input."""
        try:
            # Step 2: Preprocess
            latex = self.preprocess_latex(latex)
            
            # Step 3: Parse
            self.walker = LatexWalker(latex)
            nodes, _, _ = self.walker.get_latex_nodes()
            self.norm_str = ""
            
            # Step 4: Render
            self.build_expression(nodes, {})
            
            # Step 5: Post-process
            norm_str = self.postprocess_output(self.norm_str)
            
            # Step 6: Filter ASCII tokens
            tokens = norm_str.strip().split()
            ascii_tokens = [t for t in tokens if is_ascii(t)]
            return ' '.join(ascii_tokens), True
        except Exception as e:
            logging.error(f"Failed to parse LaTeX: {latex}, error: {str(e)}")
            return latex, False

if __name__ == "__main__":
    processor = LatexProcessor()
    latex = r"x^{2} + \text{hello}"
    result, success = processor.normalize_latex(latex)
    print(result)