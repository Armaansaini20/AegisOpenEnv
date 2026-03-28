from datasets import load_dataset

def sample_dataset():
    print("Loading SecureFinAI-Lab/Regulations_QA...")
    try:
        # We'll use a streaming load or just take the first few examples
        ds = load_dataset("SecureFinAI-Lab/Regulations_QA", split="train", streaming=True)
        iterator = iter(ds)
        print("\n--- Example 1 ---")
        item1 = next(iterator)
        print(item1)
        print("\n--- Example 2 ---")
        item2 = next(iterator)
        print(item2)
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    sample_dataset()
