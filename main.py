from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
import pandas as pd
import tempfile
import os
from openai import OpenAI
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import matplotlib.pyplot as plt

app = FastAPI()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/process-report")
async def process_report(
    brand: str = Form(...),
    month: str = Form(...),
    file: UploadFile = File(...)
):

    # SAVE XLSX
    temp_xlsx = tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx")
    temp_xlsx.write(await file.read())
    temp_xlsx.close()

    # READ EXCEL
    df = pd.read_excel(temp_xlsx.name)

    # CLEAN NUMBERS
    numeric_cols = [
        "Amount spent (USD)",
        "Reach",
        "Impressions",
        "Results"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # TOTALS
    total_spent = round(df["Amount spent (USD)"].sum(), 2)
    total_reach = int(df["Reach"].sum())
    total_impressions = int(df["Impressions"].sum())
    total_results = int(df["Results"].sum())

    # TOP ADS
    top_ads = (
        df.groupby("Ad name")["Results"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    # CREATE CHART
    chart_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name

    plt.figure(figsize=(8, 4))
    top_ads.plot(kind="bar")
    plt.title("Top Ads by Results")
    plt.tight_layout()
    plt.savefig(chart_path)

    # AI SUMMARY INPUT
    summary_prompt = f"""
    Analyze this Facebook Ads performance data.

    Brand: {brand}
    Month: {month}

    Total Spent: {total_spent}
    Total Reach: {total_reach}
    Total Impressions: {total_impressions}
    Total Results: {total_results}

    Top Ads:
    {top_ads.to_string()}

    Write:
    1. Executive summary in Georgian
    2. What worked best
    3. What underperformed
    4. Recommendations
    """

    # OPENAI CALL
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a senior paid media analyst."
            },
            {
                "role": "user",
                "content": summary_prompt
            }
        ],
        temperature=0.4
    )

    ai_summary = response.choices[0].message.content

    # CREATE PDF
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    # TITLE
    c.setFont("Helvetica-Bold", 22)
    c.drawString(50, height - 60, f"{brand} Facebook Report")

    c.setFont("Helvetica", 14)
    c.drawString(50, height - 90, f"Period: {month}")

    # KPI BLOCK
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 140, "Overview")

    c.setFont("Helvetica", 12)
    c.drawString(60, height - 170, f"Total Spent: ${total_spent}")
    c.drawString(60, height - 190, f"Total Reach: {total_reach}")
    c.drawString(60, height - 210, f"Total Impressions: {total_impressions}")
    c.drawString(60, height - 230, f"Total Results: {total_results}")

    # CHART
    c.drawImage(ImageReader(chart_path), 50, height - 500, width=500, height=220)

    # AI SUMMARY
    text = c.beginText(50, height - 540)
    text.setFont("Helvetica", 11)

    for line in ai_summary.split("\n"):
        text.textLine(line)

    c.drawText(text)

    c.save()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{brand}_{month}_report.pdf"
    )