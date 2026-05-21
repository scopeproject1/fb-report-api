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


def pct(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):,.1f}%"
    except Exception:
        return "-"


def clean_text(v):
    if pd.isna(v):
        return ""
    return str(v).strip()


def to_number(df, col):
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df


def safe_sum(df, col):
    if col not in df.columns:
        return 0
    return pd.to_numeric(df[col], errors="coerce").fillna(0).sum()


def safe_filename(text):
    text = re.sub(r"[^a-zA-Z0-9_\-]+", "_", text)
    return text.strip("_")[:80]


def dominant_text(sub, col):
    if col not in sub.columns:
        return ""
    vals = sub[col].dropna().astype(str)
    vals = vals[vals.str.strip() != ""]
    if vals.empty:
        return ""
    return vals.mode().iloc[0]


def best_breakdown(sub, col):
    if col not in sub.columns:
        return {"name": "-", "value": 0, "share": 0}

    metric = "Results" if "Results" in sub.columns and safe_sum(sub, "Results") > 0 else "Impressions"

    grouped = (
        sub.groupby(col)[metric]
        .sum()
        .sort_values(ascending=False)
    )

    if grouped.empty:
        return {"name": "-", "value": 0, "share": 0}

    total = grouped.sum()
    top_name = str(grouped.index[0])
    top_value = float(grouped.iloc[0])
    share = round((top_value / total) * 100, 1) if total else 0

    return {
        "name": top_name,
        "value": int(top_value),
        "share": share
    }


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


def build_creatives(df):
    group_cols = ["Ad name", "Campaign name", "Page name"]
    group_cols = [c for c in group_cols if c in df.columns]

    if "Ad name" not in group_cols:
        raise ValueError("Excel must contain 'Ad name' column")

    cards = []

    for keys, sub in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)

        key_map = dict(zip(group_cols, keys))

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

        result_type = dominant_text(sub, "Result type")
        objective = dominant_text(sub, "Objective")
        delivery_status = dominant_text(sub, "Delivery status")

        if not result_type:
            if messaging > 0:
                result_type = "Messaging conversations started"
            elif purchases > 0:
                result_type = "Purchases"
            elif results > 0:
                result_type = "Results"
            else:
                result_type = "No results"

        best_age = best_breakdown(sub, "Age")
        best_gender = best_breakdown(sub, "Gender")

        cards.append({
            "id": len(cards) + 1,
            "ad_name": clean_text(key_map.get("Ad name", "")),
            "campaign_name": clean_text(key_map.get("Campaign name", "")),
            "page_name": clean_text(key_map.get("Page name", "")),
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
            "raw_rows": len(sub)
        })

    cards = sorted(cards, key=lambda x: (x["results"], x["spent"]), reverse=True)

    for i, c in enumerate(cards, start=1):
        c["id"] = i

    return cards


def ai_analysis(payload):
    prompt = f"""
შენ ხარ senior Meta Ads analyst.

ქვემოთ მოცემულია Meta Ads Manager Raw XLSX export-იდან სწორად დაჯამებული მონაცემები.
გაითვალისწინე: raw rows ჩაშლილია age/gender/day breakdown-ებად, ამიტომ creative-level ანალიზი ეფუძნება aggregated creative data-ს.

მონაცემები:
{json.dumps(payload, ensure_ascii=False, indent=2)}

დაწერე ქართულად, კონკრეტულად და ბიზნესისთვის გასაგებად.

აუცილებლად გააკეთე:
1. მოკლე Executive Summary
2. მთავარი KPI შეფასება
3. საუკეთესო creatives/post-ები და რატომ იმუშავა
4. სუსტი creatives/post-ები და სავარაუდო პრობლემა
5. Audience insights — ასაკი/სქესი
6. Budget efficiency analysis
7. თითოეული მნიშვნელოვანი creative-ის მოკლე ინტერპრეტაცია
8. მომდევნო თვის action plan
9. 5 კონკრეტული რეკომენდაცია

არ გამოიგონო მონაცემები. გამოიყენე მხოლოდ მოწოდებული რიცხვები.
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
            temperature=0.3
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
        pct=pct,
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