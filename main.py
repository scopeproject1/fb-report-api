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
    return {"status": "running", "version": "media-buying-engine-v8-final-report-polish"}


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


def pct(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.1f}%"
    except Exception:
        return "-"


def roas_value(v):
    if v is None:
        return "-"
    try:
        return f"{float(v):.2f}"
    except Exception:
        return "-"


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


def safe_share(part, total):
    return round((part / total) * 100, 1) if total else 0


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


def add_share_to_breakdown(items):
    total = sum(item["value"] for item in items)
    output = []

    for item in items:
        output.append({
            "name": item["name"],
            "value": item["value"],
            "share": safe_share(item["value"], total)
        })

    return output


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

    data = add_share_to_breakdown(data)
    best = data[0]
    worst = data[-1]

    return {
        "best": {"name": best["name"], "value": best["value"], "share": best["share"]},
        "worst": {"name": worst["name"], "value": worst["value"], "share": worst["share"]},
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
            return "Good Performer", "Efficient", "Maintain budget and test variations", fatigue_risk
        if cpr is not None and cpr <= 0.10:
            return "Needs Optimization", "Moderate", "Keep limited budget and improve creative", fatigue_risk
        if spent >= 8 and cpr is not None and cpr > 0.10:
            return "Pause Candidate", "Inefficient", "Reduce budget or replace creative", fatigue_risk
        return "Needs Optimization", "Moderate", "Keep limited budget and improve creative", fatigue_risk

    if category == "MESSAGES":
        if results >= 10 and cpr is not None and cpr <= 0.40:
            return "Strong Performer", "Efficient", "Increase budget cautiously", fatigue_risk
        if results >= 4 and cpr is not None and cpr <= 0.60:
            return "Good Performer", "Balanced", "Maintain budget and test variations", fatigue_risk
        if results >= 3 and cpr is not None and cpr <= 0.90:
            return "Needs Optimization", "Expensive / Weak", "Reduce budget or improve creative", fatigue_risk
        if spent >= 3 and results >= 3 and cpr is not None and cpr > 0.90:
            return "Pause Candidate", "Inefficient", "Reduce budget or replace creative", fatigue_risk
        if spent >= 1 and results < 3:
            return "Low Data", "Insufficient Data", "Collect more data before decision", fatigue_risk
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
        summed_reach = safe_sum(sub, "Reach")
        impressions = safe_sum(sub, "Impressions")
        purchases = safe_sum(sub, "Purchases")
        messaging = safe_sum(sub, "Messaging conversations started")
        purchase_value = safe_sum(sub, "Purchases conversion value")

        cpr = round(spent / results, 4) if results else None
        cpm = round((spent / impressions) * 1000, 2) if impressions else None
        frequency = round(impressions / summed_reach, 2) if summed_reach else None
        roas = round(purchase_value / spent, 2) if purchases and purchase_value and spent else None

        objective = dominant_text(sub["Objective"]) if "Objective" in sub.columns else ""

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

            "results": int(results),
            "summed_reach": int(summed_reach),
            "reach": int(summed_reach),
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

    for i, creative in enumerate(creatives, start=1):
        creative["id"] = i

    return creatives[:MAX_CREATIVE_CARDS]


def summarize_category(items):
    total_spent = round(sum(c["spent"] for c in items), 2)
    total_results = sum(c["results"] for c in items)
    total_summed_reach = sum(c["summed_reach"] for c in items)
    total_impressions = sum(c["impressions"] for c in items)

    return {
        "count": len(items),
        "spent_usd": total_spent,
        "results": total_results,
        "summed_reach": total_summed_reach,
        "reach": total_summed_reach,
        "impressions": total_impressions,
        "avg_cpr_usd": round(total_spent / total_results, 4) if total_results else None,
        "avg_cpm_usd": round((total_spent / total_impressions) * 1000, 2) if total_impressions else None,
        "frequency": round(total_impressions / total_summed_reach, 2) if total_summed_reach else None,
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

    raw = [{"name": str(k), "value": int(v)} for k, v in grouped.items()]
    return add_share_to_breakdown(raw)


def category_items(creatives, category):
    return [c for c in creatives if c["result_category"] == category]


def top_items(items, limit=10):
    eligible = [
        c for c in items
        if c["performance_label"] not in ["Low Data", "Pause Candidate"]
    ]

    label_rank = {
        "Strong Performer": 1,
        "Good Performer": 2,
        "Needs Optimization": 3,
    }

    return sorted(
        eligible,
        key=lambda x: (
            label_rank.get(x["performance_label"], 99),
            -x["results"],
            x["cpr"] if x["cpr"] is not None else float("inf"),
            -x["spent"],
        )
    )[:limit]


def weak_items(items, limit=10):
    filtered = [
        c for c in items
        if c["performance_label"] in ["Pause Candidate", "Needs Optimization"]
    ]

    label_rank = {
        "Pause Candidate": 1,
        "Needs Optimization": 2,
    }

    return sorted(
        filtered,
        key=lambda x: (
            label_rank.get(x["performance_label"], 99),
            -(x["cpr"] if x["cpr"] is not None else 0),
            -x["spent"],
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


def label_counts(creatives):
    counts = {}
    for c in creatives:
        label = c["performance_label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def creative_concentration(creatives, total_results, category_totals):
    if not creatives or not total_results:
        return None

    top = max(creatives, key=lambda c: c["results"])
    category_results = category_totals.get(top["result_category"], {}).get("results", 0)

    return {
        "top_creative": top["ad_name"],
        "top_creative_name": top["ad_name"],
        "result_category": top["result_category"],
        "results": top["results"],
        "all_results": total_results,
        "category_results": category_results,
        "share_of_total_results": safe_share(top["results"], total_results),
        "share_of_category_results": safe_share(top["results"], category_results),
        "is_highly_concentrated": safe_share(top["results"], total_results) >= 50
    }


def ai_analysis(payload):
    prompt = f"""
You are a senior Meta Ads analyst and media buyer.

Write a concise, professional Georgian-language report based only on the JSON payload.
Keep the visible report natural and client-ready.

Important rules:
- All currency is USD. Never write GEL or "ლარი".
- Use "Summed Reach" or "Reach from breakdown rows"; do not present reach as guaranteed unique reach.
- Do not mention delivery status or use the words active, inactive, paused.
- Do not invent targeting, placements, formats, or percentages.
- Do not make behavioral audience claims such as men/women being more active or targeting should focus on a gender.
- Audience wording must describe result distribution only. Use phrasing like:
  "ამ მონაცემებში male სეგმენტზე მოდის შედეგების ყველაზე დიდი წილი."
  "მოცემულ creatives-ში female სეგმენტი ჩანს best gender-ად."
  "This is result distribution, not confirmed targeting performance."
- Low Data creatives must appear only in the Low Data Creatives section and must not be called weak, strong, underperforming, or a pause recommendation.
- Top performer sections exclude Low Data and Pause Candidate creatives.
- Needs Optimization and Pause Candidates must be discussed separately.
- Engagement and message shares must be stated separately:
  "Engagement represents X% of all results."
  "Messages represent Y% of all results."
- Creative concentration must distinguish share of all results from share inside the result category.
  If the top creative has 4,197 results, all results are 4,607, and engagement results are 4,512, write:
  "4,197 / 4,607 = 91.1% of all results" and
  "4,197 / 4,512 = 93.0% of engagement results."
  Do not say 91.1% of the engagement category.
- If ROAS is null, write exactly:
  "ROAS is not calculated because purchase data is not available."

JSON payload:
{json.dumps(payload, ensure_ascii=False, indent=2)}

Required output structure:
1. Executive Summary
2. KPI Summary
3. Result Category Analysis
4. Creative Concentration
5. Engagement Analysis
6. Message Analysis
7. Strong Performers
8. Good / Stable Performers
9. Needs Optimization
10. Pause Candidates
11. Low Data Creatives
12. Audience Distribution
13. Budget Actions
14. Next Month Action Plan
"""

    try:
        res = client.chat.completions.create(
            model=MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior paid media analyst. "
                        "Use only provided data. All currency is USD. "
                        "Do not mention delivery status. "
                        "Do not invent targeting, formats, placements, percentages, or labels. "
                        "Write naturally and concisely in Georgian."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.03
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
    total_summed_reach = int(safe_sum(df, "Reach"))
    total_impressions = int(safe_sum(df, "Impressions"))
    total_results = int(safe_sum(df, "Results"))
    total_purchases = int(safe_sum(df, "Purchases"))
    total_messaging = int(safe_sum(df, "Messaging conversations started"))
    total_purchase_value = round(safe_sum(df, "Purchases conversion value"), 2)

    avg_cpr = round(total_spent / total_results, 4) if total_results else None
    avg_cpm = round((total_spent / total_impressions) * 1000, 2) if total_impressions else None
    avg_frequency = round(total_impressions / total_summed_reach, 2) if total_summed_reach else None
    total_roas = round(total_purchase_value / total_spent, 2) if total_purchases and total_purchase_value and total_spent else None

    engagement_share_of_all_results = safe_share(category_totals["ENGAGEMENT"]["results"], total_results)
    message_share_of_all_results = safe_share(category_totals["MESSAGES"]["results"], total_results)

    weak_creatives = weak_items(creatives, 10)
    low_data_creatives = low_data_items(creatives, 10)

    strong_creatives = [
        c for c in creatives
        if c["performance_label"] == "Strong Performer"
    ]

    good_creatives = [
        c for c in creatives
        if c["performance_label"] == "Good Performer"
    ]

    needs_optimization = [
        c for c in creatives
        if c["performance_label"] == "Needs Optimization"
    ]

    pause_candidates = [
        c for c in creatives
        if c["performance_label"] == "Pause Candidate"
    ]

    category_breakdown = result_category_breakdown(creatives)
    concentration = creative_concentration(creatives, total_results, category_totals)

    payload = {
        "brand": brand,
        "month": month,
        "currency": "USD",
        "totals": {
            "spent_usd": total_spent,
            "summed_reach": total_summed_reach,
            "reach_from_breakdown_rows": total_summed_reach,
            "impressions": total_impressions,
            "all_results": total_results,
            "engagement_results": category_totals["ENGAGEMENT"]["results"],
            "message_results": category_totals["MESSAGES"]["results"],
            "engagement_share_of_all_results": engagement_share_of_all_results,
            "message_share_of_all_results": message_share_of_all_results,
            "purchases": total_purchases,
            "messaging_conversations": total_messaging,
            "purchase_value_usd": total_purchase_value,
            "avg_cpr_usd_all_results": avg_cpr,
            "avg_cost_per_engagement_usd": category_totals["ENGAGEMENT"]["avg_cpr_usd"],
            "avg_cost_per_message_usd": category_totals["MESSAGES"]["avg_cpr_usd"],
            "avg_cpm_usd": avg_cpm,
            "avg_frequency": avg_frequency,
            "roas": total_roas,
            "roas_note": "ROAS is not calculated because purchase data is not available." if total_roas is None else "",
            "raw_rows": len(df),
            "creative_count": len(creatives),
        },
        "creative_concentration": concentration,
        "label_counts": label_counts(creatives),
        "category_totals": category_totals,
        "result_category_breakdown": category_breakdown,
        "top_engagement_creatives": engagement_creatives,
        "top_message_creatives": message_creatives,
        "top_purchase_creatives": purchase_creatives,
        "strong_creatives": strong_creatives[:10],
        "good_creatives": good_creatives[:10],
        "needs_optimization": needs_optimization[:10],
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
            "summed_reach": total_summed_reach,
            "reach": total_summed_reach,
            "impressions": total_impressions,
            "results": total_results,
            "engagement_results": category_totals["ENGAGEMENT"]["results"],
            "message_results": category_totals["MESSAGES"]["results"],
            "engagement_share_of_all_results": engagement_share_of_all_results,
            "message_share_of_all_results": message_share_of_all_results,
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
            "top_creative_share": concentration["share_of_total_results"] if concentration else None,
            "top_creative_name": concentration["top_creative"] if concentration else None,
        },
        creatives=creatives,
        top_engagement_creatives=engagement_creatives,
        top_message_creatives=message_creatives,
        top_purchase_creatives=purchase_creatives,
        weak_creatives=weak_creatives,
        low_data_creatives=low_data_creatives,
        age_breakdown=payload["age_breakdown"],
        gender_breakdown=payload["gender_breakdown"],
        category_breakdown=category_breakdown,
        ai_text=ai_text,
        money=money,
        num=num,
        pct=pct,
        roas_value=roas_value,
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
