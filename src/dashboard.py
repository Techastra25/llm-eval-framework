"""
Streamlit dashboard: visualize results and let a human reviewer
confirm/override the automated judge -- the human-in-the-loop step
described in the README.

Run with:
    streamlit run src/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import streamlit as st

from src.storage import Storage

st.set_page_config(page_title="LLM Evaluation Dashboard", layout="wide")

DB_PATH = str(Path(__file__).resolve().parent.parent / "results" / "eval_results.db")
storage = Storage(db_path=DB_PATH)

st.title("LLM Evaluation & Benchmarking Dashboard")

rows = storage.fetch_all()
if not rows:
    st.warning("No results found. Run `python -m scripts.run_demo` first.")
    st.stop()

df = pd.DataFrame(rows)

# -- Summary metrics ----------------------------------------------------
st.subheader("Model Comparison")
summary = df.groupby("model_name").agg(
    avg_overall=("rubric_overall", "mean"),
    avg_correctness=("rubric_correctness", "mean"),
    avg_latency_ms=("latency_ms", "mean"),
    total_cost_usd=("cost_usd", "sum"),
    n_flagged_toxic=("toxicity_score", lambda s: (s > 0).sum()),
).round(3)
st.dataframe(summary, use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(df.groupby("model_name")["rubric_overall"].mean())
with col2:
    st.bar_chart(df.groupby("category")["rubric_overall"].mean())

# -- Human-in-the-loop review --------------------------------------------
st.subheader("Human Review Queue — Low-Scoring Responses")
st.caption("Responses the automated judge scored below 0.6. Confirm or override its call.")

low_score = df[df["rubric_overall"] < 0.6].sort_values("rubric_overall")

if low_score.empty:
    st.info("No responses currently below the review threshold.")
else:
    for _, row in low_score.iterrows():
        with st.expander(f"[{row['model_name']}] {row['prompt_id']} — judge score: {row['rubric_overall']:.2f}"):
            st.write(f"**Prediction:** {row['prediction']}")
            st.write(f"**Reference:** {row['reference']}")
            st.write(f"**Category:** {row['category']}")
            c1, c2 = st.columns(2)
            if c1.button("✅ Agree with judge", key=f"agree-{row['id']}"):
                storage.mark_human_review(int(row["id"]), agrees_with_judge=True)
                st.success("Recorded.")
            if c2.button("❌ Disagree — judge was wrong", key=f"disagree-{row['id']}"):
                storage.mark_human_review(int(row["id"]), agrees_with_judge=False)
                st.warning("Recorded as a disagreement.")

st.subheader("Full Results Table")
st.dataframe(df, use_container_width=True)
