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
MAX_CREATIVE_CARDS = 120


@app.get("/")
def root():
    return {"status": "running", "version": "aggregation-v2"}


def money(v):
    if v is None:
        return "-"
    try:
        return f"${float(v):,.2f}"
    except Exception:
        return "$0.00"


def num(v):
    if v is None:
        return "-"
    try:
        return f"{int(float(v)):,}"
    except Exception:
        return "0"


def clean_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).strip().lower()
    text = text.replace("\u200b", "").replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return text


def safe_filename(text):
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")[:80]


def to_number(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def safe_sum(df, col):
    if col not in df.columns:
        return 0
    return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()


def dominant_text(values):
    vals = pd.Series(values).dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return ""
    return vals.mode().iloc[0]


def creative_key(row):
    page = normalize_text(row.get("Page name", ""))
    campaign = normalize_text(row.get("Campaign name", ""))
    ad = normalize_text(row.get("Ad name", ""))
    return f"{page}|{campaign}|{ad}"


def sum_by_column(rows, group_col, value_col="Results"):
    result = {}

    for row in rows:
        key = clean_text(row.get(group_col, ""))
        if not key:
            continue
        value = row.get(value_col, 0) or 0
        result[key] = result.get(key, 0) + float(value)

    sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)

    return [
        {"name": str(k), "value": int(v)}
        for k, v in sorted_items
    ]


def best_segment(rows, group_col):
    data = sum_by_column(rows, group_col, "Results")

    if not data:
        data = sum_by_column(rows, group_col, "Impressions")

    if not data:
        return {"name": "-", "value": 0, "share": 0}

    total = sum(item["value"] for item in data)
    top = data[0]
    share = round((top["value"] / total) * 100, 1) if total else 0

    return {
        "name": top["name"],
        "value": top["value"],
        "share": share
    }


def build_creatives(df):
    if "Ad name" not in df.columns:
        raise ValueError("Excel must contain 'Ad name' column")

    df = df.copy()
    df["_creative_key"] = df.apply(creative_key, axis=1)

    creatives = []

    for key, sub in df.groupby("_creative_key", dropna=False):
        rows = sub.to_dict("records")

        first = rows[0]

        spent = safe_sum(sub, "Amount spent (USD)")
        results = safe_sum(sub, "Results")
        reach = safe_sum(sub, "Reach")
        impressions = safe_sum(sub, "Impressions")
        purchases = safe_sum(sub, "Purchases")
        messaging = safe_sum(sub, "Messaging conversations started")
        purchase_value = safe_sum(sub, "Purchases conversion value")

        cpr = round(spent / results, 2) if results else None
        cpm = round((spent / impressions) * 1000, 2) if impressions else None
        frequency = round(impressions / reach, 2) if reach else None
        roas = round(purchase_value / spent, 2) if spent else None

        result_type = dominant_text(sub["Result type"]) if "Result type" in sub.columns else ""
        objective = dominant_text(sub["Objective"]) if "Objective" in sub.columns else ""
        delivery_status = dominant_text(sub["Delivery status"]) if "Delivery status" in sub.columns else ""

        if not result_type:
            if messaging > 0:
                result_type = "Messaging conversations started"
            elif purchases > 0:
                result_type = "Purchases"
            elif results > 0:
                result_type = "Results"
            else:
                result_type = "No results"

        best_age = best_segment(rows, "Age")
        best_gender = best_segment(rows, "Gender")

        age_breakdown = sum_by_column(rows, "Age", "Results")
        gender_breakdown = sum_by_column(rows, "Gender", "Results")

        creatives.append({
            "ad_name": clean_text(first.get("Ad name", "")),
            "campaign_name": clean_text(first.get("Campaign name", "")),
            "page_name": clean_text(first.get("Page name", "")),
            "result_type": result_type,
            "objective": objective,
            "delivery_status": delivery_status,

            "results": int(results),
            "reach": int(reach),
            "impressions": int(impressions),
            "spent": round(float(spent), 2),

            "cpr": cpr,
            "cpm": cpm,
            "frequency": frequency,

            "purchases": int(purchases),
            "messaging": int(messaging),
            "purchase_value": round(float(purchase_value), 2),
            "roas": roas,

            "best_age": best_age,
            "best_gender": best_gender,
            "age_breakdown": age_breakdown[:5],
            "gender_breakdown": gender_breakdown[:5],

            "raw_rows": len(sub)
        })

    creatives = sorted(
        creatives,
        key=lambda x: (x["results"], x["spent"]),
        reverse=True
    )

    for i, creative in enumerate(creatives, start=1):
        creative["id"] = i

    return creatives[:MAX_CREATIVE_CARDS]


def breakdown(df, group_col, metric_col="Results", limit=10):
    if group_col not in df.columns or metric_col not in df.columns:
        return []

    temp = df.copy()
    temp[metric_col] = pd.to_numeric(temp[metric_col], errors="coerce").fillna(0)

    grouped = (
        temp.groupby(group_col)[metric_col]
        .sum()
        .sort_values(ascending=False)
        .head(limit)
    )

    return [{"name": str(k), "value": int(v)} for k, v in grouped.items()]


def ai_analysis(payload):
    prompt = f"""
შენ ხარ senior Meta Ads analyst.

მონაცემები მოდის Meta Ads Manager Raw XLSX export-იდან.
Raw rows დაყოფილია age/gender/day breakdown-ებად, მაგრამ ქვემოთ creatives უკვე სწორადაა გაერთიანებული.

მონაცემები:
{json.dumps(payload, ensure_ascii=False, indent=2)}

დაწერე ქართულად. იყავი კონკრეტული და ბიზნესისთვის სასარგებლო.

აუცილებელი წესები:
- არ გამოიგონო ისეთი რამ, რაც მონაცემებში არ ჩანს.
- არ ახსენო video/carousel/placement თუ მონაცემში არ არის.
- თუ დასკვნა არის ვარაუდი, დაწერე როგორც ვარაუდი.
- creative-level ანალიზი გააკეთე aggregated მონაცემებზე.

სტრუქტურა:
1. Executive Summary
2. KPI შეფასება
3. საუკეთესო creatives/post-ები და მიზეზები
4. სუსტი/არაეფექტური creatives/post-ები
5. Audience insights — ასაკი და სქესი
6. Budget efficiency
7. მომდევნო თვის action plan
8. 5 კონკრეტული რეკომენდაცია
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a senior paid media strategist and reporting analyst."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.25
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

    numeric_cols = [
        "Amount spent (USD)",
        "Reach",
        "Impressions",
        "Results",
        "Cost per result",
        "Purchases",
        "Cost per purchase",
        "Purchases conversion value",
        "Purchase ROAS (return on ad spend)",
        "Messaging conversations started",
        "Cost per messaging conversation started"
    ]

    for col in numeric_cols:
        df = to_number(df, col)

    creatives = build_creatives(df)

    total_spent = round(safe_sum(df, "Amount spent (USD)"), 2)
    total_reach = int(safe_sum(df, "Reach"))
    total_impressions = int(safe_sum(df, "Impressions"))
    total_results = int(safe_sum(df, "Results"))
    total_purchases = int(safe_sum(df, "Purchases"))
    total_messaging = int(safe_sum(df, "Messaging conversations started"))
    total_purchase_value = round(safe_sum(df, "Purchases conversion value"), 2)

    avg_cpr = round(total_spent / total_results, 2) if total_results else None
    avg_cpm = round((total_spent / total_impressions) * 1000, 2) if total_impressions else None
    avg_frequency = round(total_impressions / total_reach, 2) if total_reach else None
    total_roas = round(total_purchase_value / total_spent, 2) if total_spent else None

    top_creatives = creatives[:10]

    zero_result_creatives = sorted(
        [c for c in creatives if c["spent"] > 0 and c["results"] == 0],
        key=lambda x: x["spent"],
        reverse=True
    )[:10]

    high_cpr_creatives = sorted(
        [c for c in creatives if c["results"] > 0 and c["cpr"] is not None],
        key=lambda x: x["cpr"],
        reverse=True
    )[:10]

    weak_creatives = zero_result_creatives if zero_result_creatives else high_cpr_creatives

    payload = {
        "brand": brand,
        "month": month,
        "totals": {
            "spent": total_spent,
            "reach": total_reach,
            "impressions": total_impressions,
            "results": total_results,
            "purchases": total_purchases,
            "messaging_conversations": total_messaging,
            "purchase_value": total_purchase_value,
            "avg_cpr": avg_cpr,
            "avg_cpm": avg_cpm,
            "avg_frequency": avg_frequency,
            "roas": total_roas,
            "raw_rows": len(df),
            "creative_count": len(creatives),
        },
        "top_creatives": top_creatives,
        "weak_creatives": weak_creatives,
        "high_cpr_creatives": high_cpr_creatives,
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
        high_cpr_creatives=high_cpr_creatives,
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
