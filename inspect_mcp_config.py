import google.antigravity.types as ag_types
import inspect

for name in ["McpStdioServer", "McpStreamableHttpServer"]:
    if hasattr(ag_types, name):
        obj = getattr(ag_types, name)
        print(f"Name: {name}")
        print(f"Docstring: {obj.__doc__}")
        try:
            print(f"Signature: {inspect.signature(obj)}")
        except Exception as e:
            print(f"Signature error: {e}")
        print("-" * 50)
