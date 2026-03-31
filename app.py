from server.app import app

# 🏆 AegisOpenEnv: Final Submission Entry Point
# This root app.py ensures Hugging Face can find the FastAPI server 
# while satisfying the Meta OpenEnv 'multi-mode' server/ directory requirement.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
