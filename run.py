import uvicorn
from app.main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        # host="http://localhost",  # Allows external access
        # port=8080,
        reload=True,     # Auto-reload on code changes
        workers=1        # Number of worker processes
    )