from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import predict, monitor, reports, upload
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
# Trigger reload

app = FastAPI(
    title="Network IDS API",
    description="Backend API for Intrusion Detection System",
    version="1.0.0"
)

# Allow Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(predict.router, prefix="/predict", tags=["Prediction"])
app.include_router(monitor.router, prefix="/monitor", tags=["Live Monitor"])
app.include_router(reports.router, prefix="/reports", tags=["Threat Reports"])
app.include_router(upload.router, prefix="/upload", tags=["File Upload"])


@app.get("/")
def home():
    return {"message": "Network IDS Backend Running!"}
