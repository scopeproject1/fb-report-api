from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader
from playwright.async_api import async_playwright
import pandas as pd
import tempfile
import os
import json
import re
from datetime import datetime

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-4.1-mini"
MAX_CREATIVE_CARDS = 80

@app.get("/")
def root():
    return {"status": "running"}

def money(v):
    try:
        return f"${float(v):,.2f}"
    except:
        return "$0.00"

def num(v):
    try:
        return f"{int(float(v)):,}"
    except:
        return "0"

def safe_col(df, col):
    return col in df.columns

def to_number(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def clean_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()

def safe_sum(df, col):
    if col not in df.columns:
        return 0
    return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()

def safe_filename(text):
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")[:80]

def build_creatives(df):
    group_cols = []
    for col in ["Ad name", "Campaign name", "Page name", "Result type"]:
        if col in df.columns:
            group_cols.append(col)

    if not group_cols:
        group_cols = [df.columns[0]]

    agg = df.groupby(group_cols, dropna=False).agg({
        "Results": "sum" if "Results" in df.columns else "size",
        "Reach": "sum" if "Reach" in df.columns else "size",
        "Impressions": "sum" if "Impressions" in df.columns else "size",
        "Amount spent (USD)": "sum" if "Amount spent (USD)" in df.columns else "size",
    }).reset_index()

    agg = agg.rename(columns={
        "Amount spent (USD)": "spent",
        "Results": "results",
        "Reach": "reach",
        "Impressions": "impressions",
    })

    agg["cpr"] = agg.apply(lambda r: round(r["spent"] / r["results"], 2) if r["results"] else 0, axis=1)
    agg = agg.sort_values(["results", "spent"], ascending=[False, False])

    cards = []
    for i, row in agg.head(MAX_CREATIVE_CARDS).iterrows():
        cards.append({
            "id": len(cards) + 1,
            "ad_name": clean_text(row.get("Ad name", "Unknown Creative")),
            "campaign_name": clean_text(row.get("Campaign name", "")),
            "page_name": clean_text(row.get("Page name", "")),
            "result_type": clean_text(row.get("Result type", "")),
            "results": int(row.get("results", 0)),
            "reach": int(row.get("reach", 0)),
            "impressions": int(row.get("impressions", 0)),
            "spent": round(float(row.get("spent", 0)), 2),
            "cpr": row.get("cpr", 0),
        })
    return cards

def breakdown(df, group_col, metric_col="Results", limit=10):
    if group_col not in df.columns or metric_col not in df.columns:
        return []
    temp = df.copy()
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce").fillna(0)
    grouped = temp.groupby(group_col)[metric_col].sum().sort_values(ascending=False).head(limit)
    return [{"name": str(k), "value": int(v)} for k, v in grouped.items()]

def ai_analysis(payload):
    prompt = f"""
შენ ხარ senior Meta Ads analyst.

ქვემოთ მოცემულია Meta Ads Manager Raw XLSX export-იდან დაჯამებული მონაცემები.
დაწერე ქართულად, ბიზნესისთვის გასაგებად და კონკრეტულად.

მონაცემები:
{json.dumps(payload, ensure_ascii=False, indent=2)}

მომეცი:
1. Executive summary
2. მთავარი შედეგები
3. საუკეთესო creatives/post-ები და რატომ იმუშავა
4. სუსტი creatives/post-ები და რა პრობლემაა
5. Audience insights
6. Budget efficiency analysis
7. მომდევნო თვის action plan
8. 5 ძალიან კონკრეტული რეკომენდაცია
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a senior paid media strategist and reporting analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.35
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI analysis unavailable: {str(e)}"

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

    for col in ["Amount spent (USD)", "Reach", "Impressions", "Results", "Cost per result"]:
        df = to_number(df, col)

    total_spent = round(safe_sum(df, "Amount spent (USD)"), 2)
    total_reach = int(safe_sum(df, "Reach"))
    total_impressions = int(safe_sum(df, "Impressions"))
    total_results = int(safe_sum(df, "Results"))
    avg_cpr = round(total_spent / total_results, 2) if total_results else 0

    creatives = build_creatives(df)
    top_creatives = creatives[:10]
    weak_creatives = sorted(
        [c for c in creatives if c["spent"] > 0],
        key=lambda x: (x["results"], -x["spent"])
    )[:10]

    payload = {
        "brand": brand,
        "month": month,
        "totals": {
            "spent": total_spent,
            "reach": total_reach,
            "impressions": total_impressions,
            "results": total_results,
            "avg_cpr": avg_cpr,
            "rows": len(df),
            "creative_count": len(creatives),
        },
        "top_creatives": top_creatives,
        "weak_creatives": weak_creatives,
        "age_breakdown": breakdown(df, "Age"),
        "gender_breakdown": breakdown(df, "Gender"),
        "objective_breakdown": breakdown(df, "Objective"),
    }

    ai_text = ai_analysis(payload)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")

    html = template.render(
        brand=brand,
        month=month,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        totals=payload["totals"],
        creatives=creatives,
        top_creatives=top_creatives,
        weak_creatives=weak_creatives,
        age_breakdown=payload["age_breakdown"],
        gender_breakdown=payload["gender_breakdown"],
        objective_breakdown=payload["objective_breakdown"],
        ai_text=ai_text,
        money=money,
        num=num,
    )

    html_path = tempfile.NamedTemporaryFile(delete=False, suffix=".html").name
    pdf_path = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf").name

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    async with async_playwright() as p:
        browser = await p.chromium.launch(args=["--no-sandbox"])
        page = await browser.new_page()
        await page.goto(f"file://{html_path}", wait_until="networkidle")
        await page.pdf(
            path=pdf_path,
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"}
        )
        await browser.close()

    return FileResponse(
        pdf_path,
        media_type="application/pdf",
        filename=f"{safe_filename(brand)}_{month}_report.pdf"
    )