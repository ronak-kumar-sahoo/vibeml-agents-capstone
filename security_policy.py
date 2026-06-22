import ast
import os
import subprocess
import sys
import tempfile
from typing import Tuple, Dict, Any, List

# List of allowed module imports for machine learning and data science
ALLOWED_MODULES = {
    "pandas", "numpy", "sklearn", "xgboost", "matplotlib", 
    "seaborn", "joblib", "json", "math", "collections"
}

# List of blocked function calls that could be unsafe
BLOCKED_CALLS = {
    "eval", "exec", "globals", "locals", "__import__", "getattr", "setattr", "delattr"
}

class SafetyVisitor(ast.NodeVisitor):
    """
    AST visitor that audits Python code for security violations.
    Enforces a strict 'allow-list' for imports and blocks dangerous builtins.
    """
    def __init__(self):
        self.errors = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split('.')[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Import of unauthorized module '{alias.name}' is blocked.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if not node.module:
            self.errors.append("Relative imports are blocked.")
        else:
            base_module = node.module.split('.')[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Import from unauthorized module '{node.module}' is blocked.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check if calling a blocked builtin function
        if isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_CALLS:
                self.errors.append(f"Call to blocked function '{node.func.id}' is forbidden.")
        
        # Check for open() - we only allow opening files if they are in the local output directory
        # For simplicity, we block standard 'open' and force use of pandas/joblib within sandbox
        if isinstance(node.func, ast.Name) and node.func.id == "open":
            self.errors.append("Direct file open() calls are blocked. Use pandas.to_csv() or joblib.dump() to save results.")
            
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Prevent dunder attribute access like __subclasses__, __globals__, etc.
        if node.attr.startswith("__") and node.attr.endswith("__"):
            self.errors.append(f"Access to private attribute '{node.attr}' is blocked.")
        self.generic_visit(node)


def validate_code(code_str: str) -> Tuple[bool, List[str]]:
    """
    Validates Python code statically using AST analysis.
    Returns (is_safe, error_messages).
    """
    try:
        tree = ast.parse(code_str)
    except SyntaxError as e:
        return False, [f"Syntax Error: {e.msg} on line {e.lineno}"]

    visitor = SafetyVisitor()
    visitor.visit(tree)
    
    if visitor.errors:
        return False, visitor.errors
    return True, []


def execute_code_safely(code_str: str, dataset_path: str, output_dir: str) -> Dict[str, Any]:
    """
    Runs the agent's Python code inside a sandboxed subprocess.
    Passes dataset_path and output_dir as environment variables.
    """
    is_safe, errors = validate_code(code_str)
    if not is_safe:
        return {
            "success": False,
            "stdout": "",
            "stderr": "\n".join(errors),
            "error": "Security Policy Violation: The generated code failed the safety audit."
        }

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # We prepend setup code to automatically load the dataset and configure matplotlib
    # so the agent can focus on model training code.
    setup_code = f"""
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') # Force non-GUI backend for plotting
import matplotlib.pyplot as plt

DATASET_PATH = r"{dataset_path}"
OUTPUT_DIR = r"{output_dir}"

# Load dataset automatically
df = pd.read_csv(DATASET_PATH)
"""
    full_code = setup_code + "\n" + code_str

    # Write code to temporary file
    fd, temp_file_path = tempfile.mkstemp(suffix=".py", dir=output_dir)
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(full_code)

        # Run python code in restricted subprocess
        # In a real production deployment, this would run under a low-privilege user
        # or inside a Docker container. Here, we sandbox environment variables.
        env = {
            "PYTHONPATH": os.path.dirname(output_dir),
            "SYSTEMROOT": os.environ.get("SYSTEMROOT", "C:\\Windows"), # Required for Python on Windows
            "PATH": os.environ.get("PATH", "")
        }

        result = subprocess.run(
            [sys.executable, temp_file_path],
            capture_output=True,
            text=True,
            timeout=30, # Timeout to prevent infinite loops
            env=env
        )

        success = result.returncode == 0
        return {
            "success": success,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": "" if success else f"Execution failed with return code {result.returncode}"
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": "Execution timed out (limit: 30 seconds).",
            "error": "Timeout Error"
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "error": "Runtime Error"
        }
    finally:
        # Clean up temporary script
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
