import inspect
import google.antigravity as ag

def print_details(obj, name):
    print("=" * 60)
    print(f"Details for: {name}")
    print("=" * 60)
    try:
        print(f"Docstring:\n{obj.__doc__}\n")
    except Exception:
        pass
        
    try:
        sig = inspect.signature(obj)
        print(f"Signature: {sig}\n")
    except Exception as e:
        print(f"Signature error: {e}\n")
        
    try:
        print("Methods/Attributes:")
        for attr in dir(obj):
            if not attr.startswith("_"):
                print(f"  - {attr}")
    except Exception:
        pass
    print()

print_details(ag.Agent, "ag.Agent")
print_details(ag.LocalAgentConfig, "ag.LocalAgentConfig")
print_details(ag.CapabilitiesConfig, "ag.CapabilitiesConfig")
print_details(ag.BuiltinTools, "ag.BuiltinTools")
