from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routers import dataset
from .config import settings
from .utils.logger import setup_logger
import time

# Configure logging
logger = setup_logger(
    "dataset_viewer",
    "logs/dataset_viewer.log"
)

app = FastAPI(
    title="Dataset Viewer API",
    description="API for viewing datasets with audio support",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root route
@app.get("/")
async def root():
    return {
        "message": "Hello World",
        "status": "active",
        "api_version": "1.0.0",
        "documentation": "/docs"
    }

# Include routers
app.include_router(dataset.router, prefix="/api/v1")

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Log request details
    logger.info(
        f"Request started: {request.method} {request.url.path} "
        f"Client: {request.client.host}"
    )
    
    response = await call_next(request)
    
    # Calculate request duration
    duration = time.time() - start_time
    
    # Log response details
    logger.info(
        f"Request completed: {request.method} {request.url.path} "
        f"Status: {response.status_code} Duration: {duration:.3f}s"
    )
    
    return response

# Error handling
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Global error handler caught exception for request "
        f"{request.method} {request.url.path}",
        exc_info=True
    )
    
    # Log additional request details for debugging
    logger.debug(f"Request headers: {dict(request.headers)}")
    logger.debug(f"Request query params: {dict(request.query_params)}")
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )