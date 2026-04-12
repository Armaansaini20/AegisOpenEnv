import os

def check():
    path = "audit_logs.txt"
    if not os.path.exists(path):
        print("Logs not found.")
        return
        
    try:
        # PowerShell results might be in UTF-16
        with open(path, "rb") as f:
            raw = f.read()
            content = raw.decode("utf-16", "ignore")
    except:
        content = open(path, "r", errors="ignore").read()
        
    questions = [
        "What are the requirements to be listed on the Nasdaq?",
        "When was Rule. 2010"
    ]
    
    print("=== Consistency Check ===")
    for q in questions:
        found = q.lower() in content.lower()
        print(f"Found '{q}': {found}")

if __name__ == "__main__":
    check()
