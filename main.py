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
LOW_DATA_MIN_RESULTS = 3
LOW_DATA_MIN_SPEND = 1.0


@app.get("/")
def root():
    return {"status": "running", "version": "media-buying-engine-v3-refined"}


def money(v):
    if v is None:
        return "-"
    try:
        value = float(v)
        if 0 < value < 0.01:
            return "<$0.01"
        return f"${value:,.2f}"
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


def detect_result_category(row):
    result_type = normalize_text(row.get("Result type", ""))
    campaign = normalize_text(row.get("Campaign name", ""))
    objective = normalize_text(row.get("Objective", ""))

    messaging = float(row.get("Messaging conversations started", 0) or 0)
    purchases = float(row.get("Purchases", 0) or 0)

    if "messaging" in result_type or "message" in result_type:
        return "MESSAGES"
    if "post engagement" in result_type or "engagement" in result_type:
        return "ENGAGEMENT"
    if "purchase" in result_type or purchases > 0:
        return "PURCHASES"
    if "message" in campaign or "messaging" in campaign or messaging > 0:
        return "MESSAGES"
    if "engagement" in campaign or "engagement" in objective:
        return "ENGAGEMENT"

    return "OTHER"


def display_result_type(category):
    return {
        "ENGAGEMENT": "Post engagements",
        "MESSAGES": "Messaging conversations started",
        "PURCHASES": "Purchases",
        "OTHER": "Other results",
    }.get(category, "Other results")


def creative_key(row):
    page = normalize_text(row.get("Page name", ""))
    campaign = normalize_text(row.get("Campaign name", ""))
    ad = normalize_text(row.get("Ad name", ""))
    category = detect_result_category(row)
    return f"{page}|{campaign}|{ad}|{category}"


def sum_by_column(rows, group_col, value_col="Results"):
    result = {}

    for row in rows:
        key = clean_text(row.get(group_col, ""))
        if not key:
            continue

        value = row.get(value_col, 0) or 0
        result[key] = result.get(key, 0) + float(value)

    sorted_items = sorted(result.items(), key=lambda x: x[1], reverse=True)
    return [{"name": str(k), "value": int(v)} for k, v in sorted_items]


def segment_summary(rows, group_col):
    data = sum_by_column(rows, group_col, "Results")

    if not data:
        data = sum_by_column(rows, group_col, "Impressions")

    if not data:
        return {
            "best": {"name": "-", "value": 0, "share": 0},
            "worst": {"name": "-", "value": 0, "share": 0},
            "breakdown": []
        }

    total = sum(item["value"] for item in data)
    best = data[0]
    worst = data[-1]

    return {
        "best": {
            "name": best["name"],
            "value": best["value"],
            "share": round((best["value"] / total) * 100, 1) if total else 0
        },
        "worst": {
            "name": worst["name"],
            "value": worst["value"],
            "share": round((worst["value"] / total) * 100, 1) if total else 0
        },
        "breakdown": data[:5]
    }


def classify_creative(category, results, spent, cpr, cpm, frequency):
    fatigue_risk = "Low"
    if frequency is not None:
        if frequency >= 2.5:
            fatigue_risk = "High"
        elif frequency >= 1.8:
            fatigue_risk = "Medium"

    if results < LOW_DATA_MIN_RESULTS and spent < LOW_DATA_MIN_SPEND:
        return "Low Data", "Insufficient Data", "Collect more data before decision", fatigue_risk

    if category == "ENGAGEMENT":
        if results >= 1000 and cpr is not None and cpr <= 0.02:
            return "Strong Performer", "Efficient", "Increase budget cautiously", fatigue_risk
        if cpr is not None and cpr <= 0.05:
            return "Good Performer", "Efficient", "Keep active and test variations", fatigue_risk
        if cpr is not None and cpr <= 0.08:
            return "Needs Optimization", "Moderate", "Keep limited budget and improve creative", fatigue_risk
        if spent >= 3:
            return "Pause Candidate", "Inefficient", "Reduce or pause budget", fatigue_risk
        return "Low Data", "Insufficient Data", "Collect more data before decision", fatigue_risk

    if category == "MESSAGES":
        if results >= 10 and cpr is not None and cpr <= 0.40:
            return "Strong Performer", "Efficient", "Increase budget cautiously", fatigue_risk
        if results >= 4 and cpr is not None and cpr <= 0.60:
            return "Good Performer", "Balanced", "Keep stable and test variations", fatigue_risk
        if results >= 3 and cpr is not None and cpr <= 0.90:
            return "Needs Optimization", "Expensive / Weak", "Reduce budget or improve creative", fatigue_risk
        if spent >= 1 and cpr is not None and cpr > 0.90:
            return "Pause Candidate", "Inefficient", "Reduce or pause budget", fatigue_risk
        return "Low Data", "Insufficient Data", "Collect more data before decision", fatigue_risk

    if results >= 5 and cpr is not None:
        return "Needs Review", "Moderate", "Review objective-specific performance", fatigue_risk

    return "Low Data", "Insufficient Data", "Collect more data before decision", fatigue_risk


def build_creatives(df):
    if "Ad name" not in df.columns:
        raise ValueError("Excel must contain 'Ad name' column")

    df = df.copy()
    df["_result_category"] = df.apply(detect_result_category, axis=1)
    df["_creative_key"] = df.apply(creative_key, axis=1)

    creatives = []

    for _, sub in df.groupby("_creative_key", dropna=False):
        rows = sub.to_dict("records")
        first = rows[0]
        category = clean_text(first.get("_result_category", "OTHER"))

        spent = safe_sum(sub, "Amount spent (USD)")
        results = safe_sum(sub, "Results")
        reach = safe_sum(sub, "Reach")
        impressions = safe_sum(sub, "Impressions")
        purchases = safe_sum(sub, "Purchases")
        messaging = safe_sum(sub, "Messaging conversations started")
        purchase_value = safe_sum(sub, "Purchases conversion value")

        cpr = round(spent / results, 4) if results else None
        cpm = round((spent / impressions) * 1000, 2) if impressions else None
        frequency = round(impressions / reach, 2) if reach else None
        roas = round(purchase_value / spent, 2) if purchase_value and spent else None

        objective = dominant_text(sub["Objective"]) if "Objective" in sub.columns else ""
        delivery_status = dominant_text(sub["Delivery status"]) if "Delivery status" in sub.columns else ""

        age_info = segment_summary(rows, "Age")
        gender_info = segment_summary(rows, "Gender")

        label, efficiency, recommendation, fatigue_risk = classify_creative(
            category, results, spent, cpr, cpm, frequency
        )

        creatives.append({
            "ad_name": clean_text(first.get("Ad name", "")),
            "campaign_name": clean_text(first.get("Campaign name", "")),
            "page_name": clean_text(first.get("Page name", "")),

            "result_category": category,
            "result_type": display_result_type(category),
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

            "best_age": age_info["best"],
            "worst_age": age_info["worst"],
            "age_breakdown": age_info["breakdown"],

            "best_gender": gender_info["best"],
            "worst_gender": gender_info["worst"],
            "gender_breakdown": gender_info["breakdown"],

            "performance_label": label,
            "efficiency_tier": efficiency,
            "budget_recommendation": recommendation,
            "fatigue_risk": fatigue_risk,

            "raw_rows": len(sub)
        })

    creatives = sorted(
        creatives,
        key=lambda x: (x["result_category"], -x["results"], -x["spent"])
    )

    for i, creative in enumerate(creatives, start=1):
        creative["id"] = i

    return creatives[:MAX_CREATIVE_CARDS]


def summarize_category(items):
    total_spent = round(sum(c["spent"] for c in items), 2)
    total_results = sum(c["results"] for c in items)
    total_reach = sum(c["reach"] for c in items)
    total_impressions = sum(c["impressions"] for c in items)

    return {
        "count": len(items),
        "spent_usd": total_spent,
        "results": total_results,
        "reach": total_reach,
        "impressions": total_impressions,
        "avg_cpr_usd": round(total_spent / total_results, 4) if total_results else None,
        "avg_cpm_usd": round((total_spent / total_impressions) * 1000, 2) if total_impressions else None,
        "frequency": round(total_impressions / total_reach, 2) if total_reach else None,
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


def category_items(creatives, category):
    return [c for c in creatives if c["result_category"] == category]


def top_items(items, limit=10):
    label_rank = {
        "Strong Performer": 5,
        "Good Performer": 4,
        "Needs Optimization": 3,
        "Low Data": 2,
        "Pause Candidate": 1,
    }

    return sorted(
        items,
        key=lambda x: (
            label_rank.get(x["performance_label"], 0),
            x["results"],
            -(x["cpr"] or 0)
        ),
        reverse=True
    )[:limit]


def weak_items(items, limit=10):
    filtered = [
        c for c in items
        if c["performance_label"] in ["Pause Candidate", "Needs Optimization"]
        and c["performance_label"] != "Low Data"
        and c["spent"] >= 1
    ]

    label_rank = {
        "Pause Candidate": 1,
        "Needs Optimization": 2,
    }

    return sorted(
        filtered,
        key=lambda x: (
            label_rank.get(x["performance_label"], 9),
            -x["spent"],
            x["results"]
        )
    )[:limit]


def low_data_items(items, limit=10):
    return sorted(
        [c for c in items if c["performance_label"] == "Low Data"],
        key=lambda x: (-x["spent"], -x["results"])
    )[:limit]


def result_category_breakdown(creatives):
    categories = {}

    for c in creatives:
        cat = c["result_category"]
        if cat not in categories:
            categories[cat] = {
                "name": cat,
                "cards": 0,
                "results": 0,
                "spend_usd": 0
            }

        categories[cat]["cards"] += 1
        categories[cat]["results"] += c["results"]
        categories[cat]["spend_usd"] += c["spent"]

    output = []
    for item in categories.values():
        item["spend_usd"] = round(item["spend_usd"], 2)
        item["avg_cpr_usd"] = round(item["spend_usd"] / item["results"], 4) if item["results"] else None
        output.append(item)

    return sorted(output, key=lambda x: x["results"], reverse=True)


def ai_analysis(payload):
    prompt = f"""
შენ ხარ senior Meta Ads analyst და media buyer.

მონაცემები მოდის Meta Ads Manager Raw XLSX export-იდან.

მკაცრი წესები:
- ყველა monetary value არის USD-ში.
- არასოდეს გამოიყენო სიტყვა "ლარი".
- გამოიყენე მხოლოდ "USD" ან "დოლარი".
- არ გამოიგონო მონაცემი, პროცენტი, placement, format, video/carousel, behavior assumption.
- თუ მონაცემი პირდაპირ არ ჩანს payload-ში, არ დაწერო როგორც ფაქტი.
- თუ აკეთებ ვარაუდს, აუცილებლად დაწერე: "სავარაუდოდ".
- ROAS არ დაწერო 0-ად, თუ purchase_value_usd ან purchases არ არის.
- თუ purchases = 0, დაწერე: "ROAS არ არის დათვლადი, რადგან purchase data არ ფიქსირდება."
- არ შეადარო Engagement results და Message results როგორც ერთნაირი შედეგი.
- Engagement creatives შეაფასე engagement-ის ჭრილში.
- Message creatives შეაფასე cost per message / messages-ის ჭრილში.
- არ გამოიყენო ქულა/score.
- Low Data creatives-ზე არ გასცე მკაცრი pause/reduce რეკომენდაცია.
- არ გამოიყენო ზოგადი რეკომენდაცია, რომელიც მონაცემიდან არ გამომდინარეობს.

მონაცემები:
{json.dumps(payload, ensure_ascii=False, indent=2)}

დაწერე ქართულად, მოკლედ, კონკრეტულად და action-oriented.

სტრუქტურა:
1. Executive Summary
2. KPI შეფასება USD-ში
3. Engagement creatives analysis
4. Message creatives analysis
5. Strong performers
6. Pause / optimization candidates
7. Low data creatives
8. Audience insights — მხოლოდ არსებული მონაცემებით
9. Budget recommendation
10. მომდევნო თვის action plan
11. 5 კონკრეტული რეკომენდაცია
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a strict senior paid media analyst. "
                        "Use only the provided data. All currency is USD. "
                        "Never invent facts, placements, formats, percentages, or scores."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
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

    engagement_creatives_all = category_items(creatives, "ENGAGEMENT")
    message_creatives_all = category_items(creatives, "MESSAGES")
    purchase_creatives_all = category_items(creatives, "PURCHASES")

    engagement_creatives = top_items(engagement_creatives_all, 10)
    message_creatives = top_items(message_creatives_all, 10)
    purchase_creatives = top_items(purchase_creatives_all, 10)

    category_totals = {
        "ENGAGEMENT": summarize_category(engagement_creatives_all),
        "MESSAGES": summarize_category(message_creatives_all),
        "PURCHASES": summarize_category(purchase_creatives_all),
    }

    total_spent = round(safe_sum(df, "Amount spent (USD)"), 2)
    total_reach = int(safe_sum(df, "Reach"))
    total_impressions = int(safe_sum(df, "Impressions"))
    total_results = int(safe_sum(df, "Results"))
    total_purchases = int(safe_sum(df, "Purchases"))
    total_messaging = int(safe_sum(df, "Messaging conversations started"))
    total_purchase_value = round(safe_sum(df, "Purchases conversion value"), 2)

    avg_cpr = round(total_spent / total_results, 4) if total_results else None
    avg_cpm = round((total_spent / total_impressions) * 1000, 2) if total_impressions else None
    avg_frequency = round(total_impressions / total_reach, 2) if total_reach else None
    total_roas = round(total_purchase_value / total_spent, 2) if total_purchase_value and total_spent else None

    weak_creatives = weak_items(creatives, 10)
    low_data_creatives = low_data_items(creatives, 10)

    strong_creatives = [
        c for c in creatives
        if c["performance_label"] in ["Strong Performer", "Good Performer"]
    ]

    pause_candidates = [
        c for c in creatives
        if c["performance_label"] == "Pause Candidate"
    ]

    category_breakdown = result_category_breakdown(creatives)

    payload = {
        "brand": brand,
        "month": month,
        "currency": "USD",
        "totals": {
            "spent_usd": total_spent,
            "reach": total_reach,
            "impressions": total_impressions,
            "all_results": total_results,
            "engagement_results": category_totals["ENGAGEMENT"]["results"],
            "message_results": category_totals["MESSAGES"]["results"],
            "purchases": total_purchases,
            "messaging_conversations": total_messaging,
            "purchase_value_usd": total_purchase_value,
            "avg_cpr_usd_all_results": avg_cpr,
            "avg_cost_per_engagement_usd": category_totals["ENGAGEMENT"]["avg_cpr_usd"],
            "avg_cost_per_message_usd": category_totals["MESSAGES"]["avg_cpr_usd"],
            "avg_cpm_usd": avg_cpm,
            "avg_frequency": avg_frequency,
            "roas": total_roas,
            "raw_rows": len(df),
            "creative_count": len(creatives),
        },
        "category_totals": category_totals,
        "result_category_breakdown": category_breakdown,
        "top_engagement_creatives": engagement_creatives,
        "top_message_creatives": message_creatives,
        "top_purchase_creatives": purchase_creatives,
        "strong_creatives": strong_creatives[:10],
        "weak_creatives": weak_creatives,
        "pause_candidates": pause_candidates[:10],
        "low_data_creatives": low_data_creatives,
        "age_breakdown": breakdown(df, "Age"),
        "gender_breakdown": breakdown(df, "Gender"),
    }

    ai_text = ai_analysis(payload)

    env = Environment(loader=FileSystemLoader("."))
    template = env.get_template("template.html")

    html = template.render(
        brand=brand,
        month=month,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
        totals={
            "spent": total_spent,
            "reach": total_reach,
            "impressions": total_impressions,
            "results": total_results,
            "engagement_results": category_totals["ENGAGEMENT"]["results"],
            "message_results": category_totals["MESSAGES"]["results"],
            "avg_cost_per_engagement": category_totals["ENGAGEMENT"]["avg_cpr_usd"],
            "avg_cost_per_message": category_totals["MESSAGES"]["avg_cpr_usd"],
            "engagement_spend": category_totals["ENGAGEMENT"]["spent_usd"],
            "message_spend": category_totals["MESSAGES"]["spent_usd"],
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
        creatives=creatives,
        top_creatives=engagement_creatives + message_creatives + purchase_creatives,
        weak_creatives=weak_creatives,
        low_data_creatives=low_data_creatives,
        age_breakdown=payload["age_breakdown"],
        gender_breakdown=payload["gender_breakdown"],
        category_breakdown=category_breakdown,
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
