import os
import json
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

st.set_page_config(page_title="AI-Powered KPI Product Assistant", layout="wide")

st.title("AI-Powered KPI Product Assistant")
st.write("Upload KPI data, see charts, and get AI-powered product insights.")

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None

def load_data(file):
    df = pd.read_csv(file)
    if "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"])
    return df

def summarize_latest_metrics(df):
    latest = df.iloc[-1]
    previous = df.iloc[-2] if len(df) > 1 else None
    summary = {}

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in numeric_cols:
        current_val = latest[col]
        prev_val = previous[col] if previous is not None else None

        if prev_val is not None and prev_val != 0:
            pct_change = round(((current_val - prev_val) / prev_val) * 100, 2)
        else:
            pct_change = None

        summary[col] = {
            "current": current_val,
            "previous": prev_val,
            "pct_change": pct_change
        }

    return summary

def derive_product_signals(summary):
    signals = []

    retention_change = summary.get("retention_30d", {}).get("pct_change")
    churn_change = summary.get("churn_rate", {}).get("pct_change")
    conv_change = summary.get("subscription_conversion", {}).get("pct_change")
    revenue_change = summary.get("revenue", {}).get("pct_change")

    if retention_change is not None:
        if retention_change > 0:
            signals.append("Retention is improving, which means users may be finding ongoing value.")
        else:
            signals.append("Retention is declining, which may mean users are not staying engaged.")

    if churn_change is not None:
        if churn_change < 0:
            signals.append("Churn is decreasing, which is a good sign for customer stickiness.")
        else:
            signals.append("Churn is increasing, which could be a product or experience problem.")

    if conv_change is not None:
        if conv_change > 0:
            signals.append("Subscription conversion is improving, which suggests better monetization.")
        else:
            signals.append("Subscription conversion is getting weaker, which may point to pricing or value issues.")

    if revenue_change is not None:
        if revenue_change > 0:
            signals.append("Revenue is growing in the latest period.")
        else:
            signals.append("Revenue is shrinking in the latest period.")

    return signals

def call_openai_for_insights(summary, df):
    if client is None:
        return "OpenAI API key not found. Please add your key in the .env file."

    latest_rows = df.tail(4).to_dict(orient="records")
    signals = derive_product_signals(summary)

    prompt = f"""
You are a helpful Senior Product Manager.

Look at this KPI data and write:
1. Executive summary
2. Good signs
3. Risk signs
4. Possible reasons
5. Recommended next actions
6. A short summary for stakeholders

Latest KPI summary:
{json.dumps(summary, indent=2, default=str)}

Recent KPI rows:
{json.dumps(latest_rows, indent=2, default=str)}

Derived product signals:
{json.dumps(signals, indent=2)}

Use simple business language.
Be specific.
Talk about retention, churn, conversion, and revenue where useful.
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are an expert product manager and analytics strategist."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.4
    )

    return response.choices[0].message.content

st.subheader("Step 1: Upload your KPI CSV file")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)

    st.subheader("Raw KPI Data")
    st.dataframe(df, use_container_width=True)

    summary = summarize_latest_metrics(df)

    st.subheader("Latest KPI Snapshot")
    metric_names = list(summary.keys())[:8]
    cols = st.columns(4)

    for i, metric in enumerate(metric_names):
        current = summary[metric]["current"]
        delta = summary[metric]["pct_change"]

        with cols[i % 4]:
            if delta is not None:
                st.metric(label=metric, value=f"{current}", delta=f"{delta}%")
            else:
                st.metric(label=metric, value=f"{current}")

    st.subheader("Trend Chart")
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    selected_metric = st.selectbox("Choose a metric to plot", numeric_cols)

    fig = px.line(df, x="month", y=selected_metric, title=f"{selected_metric} Trend", markers=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("AI Product Insights")
    if st.button("Generate Insights"):
        with st.spinner("Thinking..."):
            insights = call_openai_for_insights(summary, df)
        st.markdown(insights)
else:
    st.info("Upload the product_kpis.csv file to begin.")