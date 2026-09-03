import streamlit as st
import requests
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from io import StringIO

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_KEY = os.getenv("API_KEY", "")

st.set_page_config(page_title="Clinical NLP — Model Arena", layout="wide", initial_sidebar_state="expanded")

# ─── Premium Glassmorphism CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f0f1a;
    --bg-secondary: #1a1a2e;
    --bg-card: rgba(26, 26, 46, 0.7);
    --accent-blue: #4f8cff;
    --accent-purple: #a855f7;
    --accent-green: #22c55e;
    --accent-amber: #f59e0b;
    --accent-red: #ef4444;
    --text-primary: #f0f0f5;
    --text-muted: #9ca3af;
    --border-glass: rgba(255,255,255,0.08);
    --glow-blue: 0 0 20px rgba(79,140,255,0.15);
}

.stApp {
    background: linear-gradient(145deg, var(--bg-primary) 0%, #16162a 50%, var(--bg-secondary) 100%);
    font-family: 'Inter', sans-serif;
    color: var(--text-primary);
}

.stSidebar {
    background: rgba(15, 15, 26, 0.85) !important;
    backdrop-filter: blur(16px);
    border-right: 1px solid var(--border-glass);
}

.stSidebar .stRadio label { font-weight: 500; }

.stButton>button {
    background: linear-gradient(135deg, var(--accent-blue) 0%, var(--accent-purple) 100%);
    color: white; border: none; border-radius: 10px; font-weight: 600;
    box-shadow: var(--glow-blue); transition: all 0.3s ease; letter-spacing: 0.02em;
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(79,140,255,0.3);
}

.stTextArea>div>div>textarea {
    background: rgba(255,255,255,0.04); border: 1px solid var(--border-glass);
    color: var(--text-primary); border-radius: 10px; font-family: 'Inter', sans-serif;
}

.stSelectbox>div>div { background: rgba(255,255,255,0.04); border-radius: 10px; }

.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.04); border-radius: 8px 8px 0 0;
    padding: 8px 16px; font-weight: 500;
}

div[data-testid="stMetric"] {
    background: var(--bg-card); backdrop-filter: blur(10px);
    border: 1px solid var(--border-glass); border-radius: 12px;
    padding: 16px; box-shadow: var(--glow-blue);
}
div[data-testid="stMetric"] label { color: var(--text-muted) !important; font-size: 0.85rem; }
div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: var(--accent-blue) !important; font-weight: 700; }

.glass-card {
    background: var(--bg-card); backdrop-filter: blur(12px);
    border: 1px solid var(--border-glass); border-radius: 14px;
    padding: 24px; margin: 12px 0; box-shadow: var(--glow-blue);
}
.glass-card h3 { color: var(--accent-blue); margin-top: 0; }

.model-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 600; letter-spacing: 0.03em; text-transform: uppercase;
}
.model-badge.clinical { background: rgba(34,197,94,0.15); color: var(--accent-green); border: 1px solid rgba(34,197,94,0.3); }
.model-badge.code { background: rgba(168,85,247,0.15); color: var(--accent-purple); border: 1px solid rgba(168,85,247,0.3); }
.model-badge.general { background: rgba(79,140,255,0.15); color: var(--accent-blue); border: 1px solid rgba(79,140,255,0.3); }

.winner-banner {
    background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(79,140,255,0.15));
    border: 1px solid rgba(34,197,94,0.3); border-radius: 12px; padding: 16px; text-align: center;
}
.winner-banner h2 { color: var(--accent-green); margin: 0; }

.score-bar { height: 8px; border-radius: 4px; background: rgba(255,255,255,0.06); overflow: hidden; margin: 4px 0; }
.score-fill { height: 100%; border-radius: 4px; transition: width 0.5s ease; }
</style>
""", unsafe_allow_html=True)


def get_headers():
    return {"X-API-Key": API_KEY, "Content-Type": "application/json"}


def fetch_models():
    """Fetch available models from API."""
    try:
        r = requests.get(f"{API_URL}/models", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"models": [], "default": "phi4-mini:latest", "evaluator": ""}


def model_selector_sidebar():
    """ChatGPT-style model selector in sidebar."""
    st.sidebar.markdown("### 🤖 Active Model")
    data = fetch_models()
    models = [m for m in data.get("models", []) if m.get("enabled", True)]

    if not models:
        st.sidebar.warning("No models found. Is Ollama running?")
        return data.get("default", "phi4-mini:latest")

    names = [m["name"] for m in models]
    displays = [f"{m.get('display_name', m['name'])}  ({m.get('size_display', '?')})" for m in models]
    default_idx = 0
    default_name = data.get("default", "")
    if default_name in names:
        default_idx = names.index(default_name)

    selected_idx = st.sidebar.selectbox("Select Model", range(len(displays)), index=default_idx,
                                        format_func=lambda i: displays[i], key="model_select")
    selected = models[selected_idx]
    cat = selected.get("category", "general")
    badge_cls = cat if cat in ("clinical", "code") else "general"
    st.sidebar.markdown(f'<span class="model-badge {badge_cls}">{cat}</span>', unsafe_allow_html=True)
    st.sidebar.caption(selected.get("description", ""))
    st.sidebar.markdown("---")
    return selected["name"]


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Process Record
# ═══════════════════════════════════════════════════════════════════════════
def process_record_page(selected_model):
    st.markdown("## 📄 Process Discharge Summary")
    st.caption(f"Active model: **{selected_model}**")

    input_text = st.text_area("Paste Discharge Summary:", height=250,
                              placeholder="Patient was admitted on...")

    if st.button("🚀 Run Pipeline", use_container_width=True):
        if not input_text:
            st.warning("Please enter text."); return

        with st.spinner(f"Running pipeline with **{selected_model}**..."):
            try:
                resp = requests.post(f"{API_URL}/process",
                    json={"text": input_text, "model_name": selected_model, "max_retries": 3},
                    headers=get_headers(), timeout=300)

                if resp.status_code == 200:
                    data = resp.json()
                    tab1, tab_onto, tab2, tab3, tab4 = st.tabs([
                        "📊 Extracted", "🎯 Ontology Grounding", "🔥 FHIR", "✅ Validation", "⏱️ Metrics"
                    ])

                    with tab1:
                        ext = data.get("extracted_data")
                        if ext:
                            if isinstance(ext, str):
                                try: ext = json.loads(ext)
                                except: ext = {"raw_text": ext}
                            st.json(ext)
                        else: st.error("Extraction failed.")

                    with tab_onto:
                        st.markdown("### 🎯 Local Medical Ontology Grounding (SNOMED CT / RxNorm / LOINC / ICD-10)")
                        st.caption("100% Local air-gapped concept mapping & confidence scores")
                        grounded = data.get("grounded_entities")
                        if grounded and isinstance(grounded, dict):
                            c1, c2 = st.columns(2)
                            with c1:
                                st.markdown("#### 🩺 Diagnoses (SNOMED CT & ICD-10)")
                                diags = grounded.get("diagnoses", [])
                                if diags:
                                    for d in diags:
                                        conf = d.get("confidence", 1.0)
                                        icd = f" | ICD-10: `{d.get('icd10_code')}`" if d.get("icd10_code") else ""
                                        st.markdown(f"• **{d.get('display')}** (SNOMED: `{d.get('code')}`{icd})")
                                        st.progress(min(max(float(conf), 0.0), 1.0), text=f"Match Confidence: {conf*100:.1f}%")
                                else:
                                    st.info("No diagnoses grounded.")

                                st.markdown("#### 💊 Medications (RxNorm)")
                                meds = grounded.get("medications", [])
                                if meds:
                                    for m in meds:
                                        conf = m.get("confidence", 1.0)
                                        dosage = f" ({m.get('dose', '')} {m.get('frequency', '')})" if m.get("dose") else ""
                                        st.markdown(f"• **{m.get('display')}**{dosage} (RxNorm: `{m.get('code')}`)")
                                        st.progress(min(max(float(conf), 0.0), 1.0), text=f"Match Confidence: {conf*100:.1f}%")
                                else:
                                    st.info("No medications grounded.")

                            with c2:
                                st.markdown("#### 🧪 Labs / Observations (LOINC)")
                                labs = grounded.get("labs", [])
                                if labs:
                                    for l in labs:
                                        conf = l.get("confidence", 1.0)
                                        val = f" = {l.get('value')} {l.get('unit','')}" if l.get("value") else ""
                                        st.markdown(f"• **{l.get('display')}**{val} (LOINC: `{l.get('code')}`)")
                                        st.progress(min(max(float(conf), 0.0), 1.0), text=f"Match Confidence: {conf*100:.1f}%")
                                else:
                                    st.info("No labs grounded.")

                                st.markdown("#### 🩺 Procedures (SNOMED CT)")
                                procs = grounded.get("procedures", [])
                                if procs:
                                    for p in procs:
                                        conf = p.get("confidence", 1.0)
                                        st.markdown(f"• **{p.get('display')}** (SNOMED: `{p.get('code')}`)")
                                        st.progress(min(max(float(conf), 0.0), 1.0), text=f"Match Confidence: {conf*100:.1f}%")
                                else:
                                    st.info("No procedures grounded.")
                        else:
                            st.info("No ontology grounding data returned.")

                    with tab2:
                        if data.get("fhir_valid"): st.success("✅ Valid FHIR Bundle")
                        else: st.error("❌ FHIR Validation Failed")
                        fb = data.get("fhir_bundle", {})
                        if isinstance(fb, str):
                            try: fb = json.loads(fb)
                            except: fb = {"raw_text": fb}
                        st.json(fb)

                    with tab3:
                        metrics = data.get("metrics", {})
                        conf = float(metrics.get("confidence", 0))
                        st.progress(conf, text=f"Confidence: {conf:.2f}")
                        val = data.get("validation_result", {})
                        for e in val.get("error_hints", []): st.warning(f"⚠️ {e}")
                        for w in val.get("warnings", []): st.info(f"ℹ️ {w}")
                        if not val.get("error_hints") and not val.get("warnings"):
                            st.success("No issues found.")

                    with tab4:
                        m = data.get("metrics", {})
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Latency", f"{float(m.get('latency_ms',0)):.0f} ms")
                        c2.metric("Retries", m.get("retry_count", 0))
                        c3.metric("Model", data.get("model_name", selected_model))
                else:
                    st.error(f"API Error: {resp.text}")
            except Exception as e:
                st.error(f"Connection failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Model Arena
# ═══════════════════════════════════════════════════════════════════════════
def model_arena_page():
    st.markdown("## ⚔️ Model Arena — Side-by-Side Comparison")
    st.caption("Run the same input through multiple models and compare results with AI evaluation.")

    data = fetch_models()
    models = [m for m in data.get("models", []) if m.get("enabled", True)]
    if len(models) < 2:
        st.warning("Need at least 2 models for arena comparison."); return

    names = [m["name"] for m in models]
    displays = {m["name"]: m.get("display_name", m["name"]) for m in models}

    selected = st.multiselect("Select models to compare", names,
                              default=names[:2],
                              format_func=lambda n: displays.get(n, n))

    if len(selected) < 2:
        st.info("Select at least 2 models."); return

    input_text = st.text_area("Paste Discharge Summary for Arena:", height=200,
                              placeholder="Patient was admitted on...", key="arena_input")
    run_eval = st.checkbox("Run AI Evaluation (uses judge model)", value=True)

    if st.button("⚔️ Start Benchmark", use_container_width=True):
        if not input_text:
            st.warning("Please enter text."); return

        with st.spinner(f"Benchmarking {len(selected)} models... This may take a few minutes."):
            try:
                resp = requests.post(f"{API_URL}/benchmark",
                    json={"text": input_text, "model_names": selected,
                          "max_retries": 3, "run_evaluation": run_eval},
                    headers=get_headers(), timeout=900)

                if resp.status_code != 200:
                    st.error(f"Benchmark failed: {resp.text}"); return

                result = resp.json()
                _render_benchmark_results(result, displays)
            except Exception as e:
                st.error(f"Benchmark error: {e}")


def _render_benchmark_results(result, displays):
    """Render benchmark results with comparison tables and charts."""
    model_results = result.get("model_results", [])
    ranking = result.get("ranking", [])

    # Winner banner
    winner = result.get("winner", "")
    if winner:
        st.markdown(f"""<div class="winner-banner">
            <h2>🏆 Winner: {displays.get(winner, winner)}</h2>
            <p>Total benchmark time: {result.get('total_time_ms', 0)/1000:.1f}s</p>
        </div>""", unsafe_allow_html=True)

    # Ranking table
    st.markdown("### 📊 Rankings")
    if ranking:
        rank_df = pd.DataFrame(ranking)
        rank_df.index = range(1, len(rank_df) + 1)
        rank_df.index.name = "Rank"
        rank_df.columns = ["Model", "Overall Score", "Latency (ms)", "FHIR Valid", "Confidence"]
        st.dataframe(rank_df, use_container_width=True)

    # Radar chart
    if any(mr.get("evaluation") for mr in model_results):
        st.markdown("### 🕸️ Evaluation Radar")
        cats = ["Accuracy", "Completeness", "FHIR Compliance", "Format Quality"]
        fig = go.Figure()
        colors = ["#4f8cff", "#a855f7", "#22c55e", "#f59e0b", "#ef4444"]
        for i, mr in enumerate(model_results):
            ev = mr.get("evaluation", {})
            if ev:
                vals = [ev.get("accuracy", 0), ev.get("completeness", 0),
                        ev.get("fhir_compliance", 0), ev.get("format_quality", 0)]
                fig.add_trace(go.Scatterpolar(r=vals + [vals[0]], theta=cats + [cats[0]],
                    fill='toself', name=displays.get(mr["model_name"], mr["model_name"]),
                    line=dict(color=colors[i % len(colors)]),
                    fillcolor=colors[i % len(colors)].replace(")", ",0.1)").replace("rgb", "rgba") if "rgb" in colors[i % len(colors)] else None))
        fig.update_layout(polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.1)")),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#f0f0f5", family="Inter"), showlegend=True,
            legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=30, b=30))
        st.plotly_chart(fig, use_container_width=True)

    # Side-by-side extracted data
    st.markdown("### 🔍 Extracted Data Comparison")
    cols = st.columns(len(model_results))
    for i, mr in enumerate(model_results):
        with cols[i]:
            name = displays.get(mr["model_name"], mr["model_name"])
            st.markdown(f"**{name}**")
            if mr.get("error"):
                st.error(mr["error"])
            elif mr.get("extracted_data"):
                ext = mr["extracted_data"]
                if isinstance(ext, str):
                    try: ext = json.loads(ext)
                    except: ext = {"raw_text": ext}
                st.json(ext)
            else:
                st.warning("No data extracted")

    # Evaluation details
    if any(mr.get("evaluation") for mr in model_results):
        st.markdown("### 📝 Evaluation Details")
        for mr in model_results:
            ev = mr.get("evaluation", {})
            if not ev: continue
            name = displays.get(mr["model_name"], mr["model_name"])
            with st.expander(f"{name} — Score: {ev.get('overall_score', 0)}/100"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy", f"{ev.get('accuracy', 0)}")
                c2.metric("Completeness", f"{ev.get('completeness', 0)}")
                c3.metric("FHIR", f"{ev.get('fhir_compliance', 0)}")
                c4.metric("Format", f"{ev.get('format_quality', 0)}")
                reasoning = ev.get("reasoning", {})
                if reasoning:
                    for dim, text in reasoning.items():
                        st.caption(f"**{dim}**: {text}")
                hall = ev.get("hallucinations_found", [])
                missed = ev.get("missed_entities", [])
                if hall: st.warning(f"Hallucinations: {', '.join(str(h) for h in hall)}")
                if missed: st.info(f"Missed: {', '.join(str(m) for m in missed)}")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Leaderboard
# ═══════════════════════════════════════════════════════════════════════════
def leaderboard_page():
    st.markdown("## 🏆 Model Leaderboard")
    st.caption("Aggregated performance rankings — updated in real-time after each run.")

    try:
        resp = requests.get(f"{API_URL}/leaderboard", timeout=10)
        if resp.status_code != 200:
            st.warning("Could not load leaderboard."); return
        data = resp.json()
    except Exception:
        st.warning("API unavailable."); return

    lb = data.get("leaderboard", [])
    total = data.get("total_runs", 0)
    st.caption(f"Total recorded runs: **{total}**")

    if not lb:
        st.info("No model data yet. Process some records first!"); return

    # Summary metrics
    cols = st.columns(len(lb))
    medals = ["🥇", "🥈", "🥉"] + ["🏅"] * 10
    for i, entry in enumerate(lb):
        with cols[i]:
            st.markdown(f"### {medals[i]} {entry['model_name']}")
            st.metric("Runs", entry["total_runs"])
            st.metric("Avg Latency", f"{entry['avg_latency_ms']:.0f} ms")
            st.metric("FHIR Success", f"{entry['fhir_success_rate']:.0f}%")
            if entry.get("avg_overall_score") is not None:
                st.metric("Eval Score", f"{entry['avg_overall_score']:.1f}/100")

    # Bar chart comparison
    st.markdown("### 📊 Performance Comparison")
    df = pd.DataFrame(lb)
    score_cols = [c for c in ["avg_accuracy", "avg_completeness", "avg_fhir_score",
                              "avg_format_score"] if c in df.columns and df[c].notna().any()]

    if score_cols:
        melt_df = df.melt(id_vars=["model_name"], value_vars=score_cols,
                          var_name="Metric", value_name="Score")
        melt_df["Metric"] = melt_df["Metric"].str.replace("avg_", "").str.replace("_", " ").str.title()
        fig = px.bar(melt_df, x="model_name", y="Score", color="Metric", barmode="group",
                     color_discrete_sequence=["#4f8cff", "#a855f7", "#22c55e", "#f59e0b"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#f0f0f5", family="Inter"),
                          xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                          yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[0, 100]),
                          legend=dict(bgcolor="rgba(0,0,0,0)"), margin=dict(t=30))
        st.plotly_chart(fig, use_container_width=True)

    # Latency comparison
    st.markdown("### ⏱️ Latency Comparison")
    fig2 = px.bar(df, x="model_name", y="avg_latency_ms",
                  color_discrete_sequence=["#4f8cff"])
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#f0f0f5", family="Inter"),
                       yaxis=dict(title="Avg Latency (ms)", gridcolor="rgba(255,255,255,0.08)"),
                       xaxis=dict(title=""), margin=dict(t=30))
    st.plotly_chart(fig2, use_container_width=True)

    # Full leaderboard table
    st.markdown("### 📋 Full Rankings Table")
    display_df = pd.DataFrame(lb)
    st.dataframe(display_df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Benchmark History
# ═══════════════════════════════════════════════════════════════════════════
def benchmark_history_page():
    st.markdown("## 📈 Benchmark History")

    try:
        resp = requests.get(f"{API_URL}/metrics/export", headers=get_headers(), timeout=10)
        if resp.status_code != 200:
            st.warning("No metrics found."); return
        df = pd.read_csv(StringIO(resp.text))
    except Exception:
        st.warning("Could not load metrics."); return

    if df.empty:
        st.info("No data yet."); return

    # Model filter
    if "model_name" in df.columns:
        models_in_data = df["model_name"].dropna().unique().tolist()
        if models_in_data:
            filter_model = st.selectbox("Filter by model", ["All"] + models_in_data)
            if filter_model != "All":
                df = df[df["model_name"] == filter_model]

    st.dataframe(df, use_container_width=True)

    # Trend chart
    if "timestamp" in df.columns and "model_name" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df_valid = df.dropna(subset=["timestamp"])
        if not df_valid.empty and "confidence" in df_valid.columns:
            st.markdown("### 📈 Confidence Over Time")
            fig = px.line(df_valid, x="timestamp", y="confidence", color="model_name",
                         color_discrete_sequence=["#4f8cff", "#a855f7", "#22c55e", "#f59e0b"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                             font=dict(color="#f0f0f5"), yaxis=dict(range=[0, 1.05],
                             gridcolor="rgba(255,255,255,0.08)"), margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

    st.download_button("⬇️ Download CSV", df.to_csv(index=False), "benchmark_history.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: Model Management
# ═══════════════════════════════════════════════════════════════════════════
def model_management_page():
    st.markdown("## ⚙️ Model Management")
    st.caption("View, configure, and add Ollama models to the pipeline.")

    data = fetch_models()
    models = data.get("models", [])

    st.markdown(f"**Default model**: `{data.get('default', 'N/A')}`")
    st.markdown(f"**Evaluator model**: `{data.get('evaluator', 'N/A')}`")

    # Models table
    st.markdown("### 📦 Installed Models")
    if models:
        for m in models:
            cat = m.get("category", "general")
            badge_cls = cat if cat in ("clinical", "code") else "general"
            with st.expander(f"{'✅' if m.get('enabled') else '❌'} {m.get('display_name', m['name'])} — {m.get('size_display', '?')}"):
                st.markdown(f'<span class="model-badge {badge_cls}">{cat}</span>', unsafe_allow_html=True)
                st.caption(m.get("description", ""))
                c1, c2, c3 = st.columns(3)
                c1.write(f"**Name:** `{m['name']}`")
                c2.write(f"**Params:** {m.get('parameter_count', 'Unknown')}")
                c3.write(f"**Size:** {m.get('size_display', 'Unknown')}")

                # Toggle
                new_state = st.checkbox("Enabled", value=m.get("enabled", True), key=f"toggle_{m['name']}")
                if new_state != m.get("enabled", True):
                    try:
                        requests.post(f"{API_URL}/models/{m['name']}/toggle",
                            json={"enabled": new_state}, headers=get_headers(), timeout=5)
                        st.success(f"{'Enabled' if new_state else 'Disabled'} {m['name']}")
                        st.rerun()
                    except Exception:
                        st.error("Failed to update.")
    else:
        st.warning("No models found. Ensure Ollama is running.")

    # Add model form
    st.markdown("### ➕ Register New Model")
    st.info("First install via `ollama pull <model-name>`, then register it here.")

    with st.form("add_model_form"):
        name = st.text_input("Model name (as shown in `ollama list`)", placeholder="llama3.2:3b")
        display = st.text_input("Display name", placeholder="Llama 3.2 3B")
        category = st.selectbox("Category", ["general", "clinical", "code"])
        desc = st.text_area("Description", placeholder="Brief description of the model...", height=80)
        params = st.text_input("Parameter count", placeholder="3B")

        if st.form_submit_button("Register Model"):
            if not name:
                st.error("Model name is required."); return
            try:
                resp = requests.post(f"{API_URL}/models/add",
                    json={"name": name, "display_name": display, "category": category,
                          "description": desc, "parameter_count": params},
                    headers=get_headers(), timeout=10)
                result = resp.json()
                if result.get("status") == "ok":
                    st.success(result["message"])
                    st.rerun()
                else:
                    st.error(result.get("message", "Failed"))
            except Exception as e:
                st.error(f"Error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    st.sidebar.markdown("""
    <div style="text-align:center; padding: 10px 0 20px 0;">
        <span style="font-size: 2rem;">🧬</span><br>
        <span style="font-size: 1.1rem; font-weight: 700; letter-spacing: 0.05em;">
            Clinical NLP Arena
        </span><br>
        <span style="font-size: 0.75rem; color: #9ca3af;">Multi-Model Benchmarking</span>
    </div>
    """, unsafe_allow_html=True)

    selected_model = model_selector_sidebar()

    page = st.sidebar.radio("Navigation", [
        "📄 Process Record",
        "⚔️ Model Arena",
        "🏆 Leaderboard",
        "📈 Benchmark History",
        "⚙️ Model Management",
    ])

    st.sidebar.markdown("---")
    st.sidebar.caption("Fully Local • Privacy Preserving • Multi-Model")

    if page == "📄 Process Record":
        process_record_page(selected_model)
    elif page == "⚔️ Model Arena":
        model_arena_page()
    elif page == "🏆 Leaderboard":
        leaderboard_page()
    elif page == "📈 Benchmark History":
        benchmark_history_page()
    elif page == "⚙️ Model Management":
        model_management_page()


if __name__ == "__main__":
    main()
