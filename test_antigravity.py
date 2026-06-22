import sys
try:
    import google.antigravity as ag
    print("SUCCESS: google.antigravity imported")
    print("Attributes in google.antigravity:")
    print(dir(ag))
    if hasattr(ag, "Agent"):
        print("Agent class is present")
    if hasattr(ag, "LocalAgentConfig"):
        print("LocalAgentConfig is present")
except Exception as e:
    print(f"FAILED: {e}")
