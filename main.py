from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ReportRequest(BaseModel):
    brand: str
    month: str
    drive_file_id: str
    file_name: str
    telegram_chat_id: str

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/process-report")
def process_report(data: ReportRequest):
    return {
        "status": "success",
        "brand": data.brand,
        "month": data.month,
        "file": data.file_name,
        "drive_file_id": data.drive_file_id
    }