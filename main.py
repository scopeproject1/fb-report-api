from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

app = FastAPI()

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/process-report")
async def process_report(
    brand: str = Form(...),
    month: str = Form(...),
    file: UploadFile = File(...)
):
    temp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    temp_xlsx.write(await file.read())
    temp_xlsx.close()

    df = pd.read_excel(temp_xlsx.name)

    rows = len(df)
    columns = list(df.columns)

    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, f"{brand} - {month} Facebook Ads Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 100, f"Rows parsed: {rows}")
    c.drawString(50, height - 125, f"Columns detected: {len(columns)}")

    y = height - 170
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y, "Detected columns:")
    y -= 25

    c.setFont("Helvetica", 9)
    for col in columns[:35]:
        c.drawString(60, y, f"- {col}")
        y -= 14
        if y < 60:
            c.showPage()
            y = height - 60

    c.save()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{brand}_{month}_report.pdf"
    )