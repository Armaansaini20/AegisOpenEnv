from datasets import load_dataset
import os

def verify():
    log_path = "audit_logs.txt"
    if not os.path.exists(log_path):
        print("Logs not found.")
        return
        
    # Read logs (UTF-16 from PowerShell redirection)
    with open(log_path, "rb") as f:
        content = f.read().decode("utf-16", "ignore")
        
    print("--- Loading Dataset ---")
    ds = load_dataset("SecureFinAI-Lab/Regulations_QA", split="train", streaming=True)
    it = iter(ds)
    
    print("--- Comparing Samples ---")
    for i in range(5):
        sample = next(it)
        q = sample["question"]
        found = q in content
        print(f"Sample {i+1}: '{q[:50]}...' Found: {found}")

if __name__ == "__main__":
    verify()
