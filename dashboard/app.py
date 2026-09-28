#!/usr/bin/env python3
"""
Financial News Intelligence Dashboard - Company Centric View
AI-powered financial news company tagging
"""

import json
import streamlit as st
from datetime import datetime

st.set_page_config(page_title="Company Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .reportview-container { background-color: #f5f7fa; padding: 2px; }
    .header-bar { display: flex; justify-content: space-between; align-items: center;
        padding: 2px 3px; background-color: #2c3e50; border-radius: 2px; margin-bottom: 2px; }
    .header-title { color: white; font-size: 18px; font-weight: bold; }
    .metric-box { background-color: #3498db; color: white; border-radius: 2px;
        padding: 2px 3px; text-align: center; margin-right: 4px; min-width: 80px; }
    .metric-value { font-size: 20px; font-weight: bold; }
    .metric-label { font-size: 11px; opacity: 0.9; }
    .article-detail-panel { border: 1px solid #ddd; border-radius: 2px; padding: 0px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .article-detail-header { font-size: 16px; font-weight: bold; color: #2c3e50; margin-bottom: 2px; border-bottom: 2px solid #3498db; padding-bottom: 2px; }
    .filter-row { display: flex; gap: 2px; align-items: flex-start; background-color: white; padding: 4px; border-radius: 2px; margin: 2px 0 2px 0; border: 1px solid #e0e0e0; }
    .filter-group { display: flex; flex-direction: column; gap: 2px; min-width: 120px; }
    .filter-label { font-size: 11px; font-weight: bold; color: #555; margin-bottom: 2px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_articles():
    with open("data/batch_results.json", "r") as f:
        raw = json.load(f)
    raw_by_id = {a["id"]: a for a in load_raw_articles()}

    normalized = []
    for article in raw:
        pub_date = raw_by_id.get(article["article_id"], {}).get("published_date", "")
        predictions = []
        for tag in article.get("tags", []):
            predictions.append({
                "company_name": tag.get("company"),
                "relationship_level": tag.get("relation"),
                "confidence_label": tag.get("confidence"),  # "high"/"medium"/"low"
                "evidence": tag.get("rationale", tag.get("reason", "")),
                "role": tag.get("role"),
                "timestamp": pub_date,
            })
        normalized.append({
            "article_id": article["article_id"],
            "title": article["title"],
            "predictions": predictions,
        })
    return normalized

@st.cache_data
def load_companies():
    with open("data/companies.json", "r") as f:
        return json.load(f)  # already a flat list — no "profiles" key to unwrap

@st.cache_data
def load_raw_articles():
    with open("data/articles.json", "r") as f:
        return json.load(f)  # uses "id", not "article_id"

articles = load_articles()
company_profiles = load_companies()
raw_articles = load_raw_articles()

company_map = {c["name"]: c for c in company_profiles}
company_names = sorted([c["name"] for c in company_profiles])

col1, col2 = st.columns([3, 1], gap='xsmall')
with col1:
    st.markdown('<div class="header-bar"><span class="header-title">COMPANY INTELLIGENCE</span></div>', unsafe_allow_html=True)

with col2:
    selected_company_name = st.selectbox(label="Company", options=company_names, label_visibility="collapsed", key="company_selector")

if not articles or not company_profiles:
    st.warning("No data available.")
    st.stop()

selected_company = company_map.get(selected_company_name, {})
st.markdown("---")
comp_col1, comp_col2, comp_col3 = st.columns(3)
with comp_col1:
    name = selected_company.get("name", "N/A")
    st.write(f"**{name}**")
with comp_col2:
    sector = selected_company.get("sector", "N/A")
    if sector: st.write(f"Sector: {sector}")
with comp_col3:
    pass  # reserved — no ticker/website data available yet
st.caption(selected_company.get("description", ""))

# Filter controls - placed before metrics
st.markdown("---")
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
with filter_col1:
    st.markdown('<div class="filter-group"><span class="filter-label">Relationship Level</span></div>', unsafe_allow_html=True)
    relationship_filter = st.selectbox("", ["All", "L1", "L2"], label_visibility="collapsed", key="rel_filter")
with filter_col2:
    st.markdown('<div class="filter-group"><span class="filter-label">Confidence</span></div>', unsafe_allow_html=True)
    confidence_filter = st.selectbox("", ["All", "High only", "High + Medium"], label_visibility="collapsed", key="conf_filter")
with filter_col3:
    st.markdown('<div class="filter-group"><span class="filter-label">Date Range</span></div>', unsafe_allow_html=True)
    date_filter = st.selectbox("", ["All", "Last 7 days", "Last 30 days", "Last 90 days"], label_visibility="collapsed", key="date_filter")
with filter_col4:
    st.markdown('<div class="filter-group"><span class="filter-label">Headline Search</span></div>', unsafe_allow_html=True)
    headline_search = st.text_input("", "", label_visibility="collapsed", key="search_filter")

tagged_articles = []
l1_count = l2_count = 0

for article in articles:
    for pred in article.get("predictions", []):
        if pred.get("company_name") == selected_company_name:
            tagged_articles.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "level": pred.get("relationship_level", ""),
                "confidence": pred.get("confidence_label", ""),
                "role": pred.get("role", ""),
                "evidence": pred.get("evidence", ""),
                "timestamp": pred.get("timestamp", "")
            })
            if pred.get("relationship_level") == "L1": l1_count += 1
            elif pred.get("relationship_level") == "L2": l2_count += 1

# Calculate ALL metrics for table display
total_all = len(tagged_articles)
l1_all = l1_count  # Already calculated above
l2_all = l2_count

# Calculate date range reference: find the latest article date in tagged_articles
latest_date = None
for item in tagged_articles:
    ts = item.get("timestamp", "")
    if ts and len(ts) >= 10:
        try:
            article_date = datetime.strptime(ts[:10], "%Y-%m-%d")
            if latest_date is None or article_date > latest_date:
                latest_date = article_date
        except (ValueError, TypeError):
            pass

# Apply filters to tagged_articles
filtered_articles = []
if latest_date:
    date_thresholds = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
    for article in tagged_articles:
        # Filter by relationship level
        if relationship_filter != "All" and article.get("level") != relationship_filter:
            continue

        # Filter by confidence
        if confidence_filter == "High only" and article.get("confidence") != "high":
            continue
        if confidence_filter == "High + Medium" and article.get("confidence") not in ("high", "medium"):
            continue

        # Filter by date range (relative to latest article date)
        if date_filter != "All":
            ts = article.get("timestamp", "")
            if ts and len(ts) >= 10:
                try:
                    article_date = datetime.strptime(ts[:10], "%Y-%m-%d")
                    days_diff = (latest_date - article_date).days
                    if days_diff > date_thresholds[date_filter]:
                        continue
                except (ValueError, TypeError):
                    pass

        # Filter by headline search
        if headline_search:
            headline = article.get("title", "").lower()
            search_term = headline_search.lower()
            if search_term not in headline:
                continue

        filtered_articles.append(article)

# Calculate FILTERED metrics for table display
total_filtered = len(filtered_articles)
l1_filtered = len([a for a in filtered_articles if a.get("level") == "L1"])
l2_filtered = len([a for a in filtered_articles if a.get("level") == "L2"])

# Display compact tagging summary table using st.dataframe

# Calculate total for both sets
total_all_calc = l1_all + l2_all
total_filtered_calc = l1_filtered + l2_filtered

tagging_data = [
    {"Scope": "Total", "Tagged News": total_all_calc, "L1 (Direct Exposure)": l1_all, "L2 (Predicted Relationship)": l2_all},
    {"Scope": "Filtered", "Tagged News": total_filtered_calc, "L1 (Direct Exposure)": l1_filtered, "L2 (Predicted Relationship)": l2_filtered}
]

st.dataframe(
    tagging_data,
    # use_container_width=True,
    width=700, 
    hide_index=True,
    column_config={
        "Scope": st.column_config.TextColumn("Scope", width= 100), #"small"),
        "Tagged News": st.column_config.NumberColumn("Tagged News", width= 100), #"small"),
        "L1 (Direct Exposure)": st.column_config.NumberColumn("L1 (Direct Exposure)", width= 100), #"small"),
        "L2 (Predicted Relationship)": st.column_config.NumberColumn("L2 (Predicted Relationship)", width= 100), #"small")
    },
    key="tagging_summary_df"
)
st.subheader(f"NEWS AFFECTING {selected_company_name.upper()}")
if filtered_articles:
    table_data = []
    for item in filtered_articles:
        row = {"Date": item["timestamp"][:10] if item.get("timestamp") else "",
            "Headline": item["title"], "Level": item["level"], "Confidence": item["confidence"].capitalize() if item["confidence"] else "N/A",
            "article_id": item["article_id"]}
        table_data.append(row)
    level_order = {"L1": 1, "L2": 2}
    table_data.sort(key=lambda x: (x["Date"], level_order.get(x["Level"], 99)), reverse=True)
    if table_data:
        selected = st.dataframe(table_data, use_container_width=True, hide_index=True,
            column_config={
                "article_id": None,
                "Date": st.column_config.TextColumn("Date", width="medium"),
                "Headline": st.column_config.TextColumn("Headline", width="large"),
                "Level": st.column_config.TextColumn("Level", width="small"),
                "Confidence": st.column_config.TextColumn("Confidence", width="small")
            },
            on_select="rerun",
            selection_mode="single-row")
        
        selected_row = None
        if "selection" in selected:
            selection_data = selected["selection"]
            if isinstance(selection_data, dict) and "rows" in selection_data:
                selected_indices = selection_data.get("rows", [])
                if selected_indices:
                    selected_row = table_data[selected_indices[0]]
        
        current_row = selected_row if selected_row else (table_data[0] if table_data else {})
        
        article_id = current_row.get("article_id")
        raw_article = None
        if article_id:
            for a in raw_articles:
                if a.get("id") == article_id:
                    raw_article = a
                    break
        
        st.markdown('<div class="article-detail-panel">', unsafe_allow_html=True)
        st.markdown('<div class="article-detail-header">Article Details</div>', unsafe_allow_html=True)
        
        st.write(f"**Date**: {current_row.get('Date', 'N/A')}")
        st.write(f"**Headline**: {current_row.get('Headline', 'N/A')}")
        
        if raw_article:
            source = raw_article.get("source", "")
            if source:
                st.write(f"**Source**: {source}")
        
        st.write(f"**Relationship Level**: {current_row.get('Level', 'N/A')}")
        st.write(f"**Confidence**: {current_row.get('Confidence', 'N/A')}")
        
        for item in filtered_articles:
            if item["title"] == current_row.get("Headline"):
                if item.get("evidence"):
                    st.write(f"**Evidence**: {item['evidence'][:200]}...")
                break
        
        st.markdown("---")
        
        if raw_article and raw_article.get("content"):
            st.write("**Full Article Text:**")
            st.text(raw_article.get("content", ""))
        else:
            st.write("Article text unavailable")
        
        st.markdown('</div>', unsafe_allow_html=True)
    else: st.info("No news matches the selected filters.")
else: st.info("No news matches the selected filters.")
st.markdown("---")
st.caption(f"Company Intelligence Dashboard - Showing data for: {selected_company_name}")
