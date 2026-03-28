import argparse
import subprocess
import sys
import os

def check_huggingface_login():
    """Verify the user is logged into Hugging Face CLI."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.whoami()
        return True
    except Exception:
        return False

def init_space(space_name: str, private: bool = True):
    """Initialize a Hugging Face space for the environment."""
    print(f"--- Initializing Hugging Face Space: {space_name} ---")
    
    if not check_huggingface_login():
        print("Error: You are not logged in to Hugging Face CLI. Please run 'huggingface-cli login' first.")
        sys.exit(1)
        
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=space_name, repo_type="space", space_sdk="docker", private=private, exist_ok=True)
        print("Space created or verified successfully!")
    except Exception as e:
        print(f"Failed to create space: {e}")
        print("Note: If it already exists, deployment will proceed.")

def deploy_to_space(space_name: str):
    """Deploy the current directory to the Hugging Face Space."""
    print(f"\n--- Deploying AegisGym to {space_name} ---")
    
    # Normally this would involve git add/commit/push to the HF remote
    # or using the huggingface_hub Python library to upload the folder seamlessly.
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        print("Uploading files to Hub...")
        
        # Don't upload the cache or local virtual environments
        ignore_patterns = ["__pycache__/*", "*.git*", ".env", "venv/*"]
        
        url = api.upload_folder(
            folder_path=".",
            repo_id=space_name,
            repo_type="space",
            ignore_patterns=ignore_patterns
        )
        print(f"Deployment successful! Your environment is live at: {url}")
        
    except ImportError:
        print("Error: 'huggingface_hub' is not installed. Run 'pip install huggingface_hub'.")
        sys.exit(1)
    except Exception as e:
        print(f"Deployment failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy AegisGym to Hugging Face Spaces.")
    parser.add_argument("space_name", help="The name of your huggingface space (e.g. username/aegisgym)")
    parser.add_argument("--public", action="store_true", help="Make the space public (default is private)")
    
    args = parser.parse_args()
    
    init_space(args.space_name, private=not args.public)
    deploy_to_space(args.space_name)
    
    print("\n--- Next Steps ---")
    print("1. Your Space is building the Docker container using your OpenEnv configurations.")
    print(f"2. You can access it directly via: https://huggingface.co/spaces/{args.space_name}")
    print("3. In your training script, switch your Env URL to match your Space.")
