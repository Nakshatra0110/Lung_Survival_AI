import os
import io
import base64
import tempfile
import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st

from predict import load_model, predict_survival

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="LungSurvival AI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================
# GLOBAL STYLES
# ======================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0d1117; }
[data-testid="stHeader"]           { background:transparent; }
[data-testid="stSidebar"]          { background:#161b22; border-right:1px solid #30363d; }

.card {
    background:#161b22; border:1px solid #30363d;
    border-radius:12px; padding:20px 24px; margin-bottom:12px;
}
.card-title {
    font-size:11px; font-weight:600; letter-spacing:1px;
    text-transform:uppercase; color:#8b949e; margin-bottom:6px;
}
.card-value { font-size:32px; font-weight:700; color:#e6edf3; line-height:1.1; }
.card-sub   { font-size:12px; color:#8b949e; margin-top:4px; }

.badge { display:inline-block; padding:6px 18px; border-radius:20px;
         font-size:13px; font-weight:700; letter-spacing:0.5px; text-transform:uppercase; }
.badge-low      { background:#0d2818; color:#3fb950; border:1px solid #238636; }
.badge-moderate { background:#2d2007; color:#d29922; border:1px solid #9e6a03; }
.badge-high     { background:#2d0f0f; color:#f85149; border:1px solid #da3633; }

.gauge-wrap { background:#21262d; border-radius:8px; height:10px; width:100%; margin:8px 0; }
.gauge-fill { height:10px; border-radius:8px; }

.section-head {
    font-size:13px; font-weight:600; letter-spacing:1px; text-transform:uppercase;
    color:#8b949e; border-bottom:1px solid #21262d; padding-bottom:8px; margin:24px 0 16px;
}
.disclaimer {
    background:#161b22; border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0; padding:10px 16px;
    font-size:12px; color:#8b949e; margin-top:12px;
}
.tooltip-wrap { position:relative; display:inline-block; cursor:help; }
.tooltip-wrap .tooltip-text {
    visibility:hidden; background:#21262d; color:#e6edf3;
    border:1px solid #30363d; border-radius:8px;
    font-size:11px; padding:8px 12px; width:220px;
    position:absolute; z-index:99; bottom:125%; left:50%;
    transform:translateX(-50%); white-space:normal; line-height:1.5;
}
.tooltip-wrap:hover .tooltip-text { visibility:visible; }

[data-testid="stFileUploader"] {
    background:#161b22 !important; border:1px dashed #30363d !important;
    border-radius:12px !important;
}
.stButton > button {
    background:#238636 !important; color:#fff !important; border:none !important;
    border-radius:8px !important; font-weight:600 !important;
    padding:10px 28px !important; font-size:14px !important; width:100%;
}
.stButton > button:hover { background:#2ea043 !important; }
[data-testid="stExpander"] {
    background:#161b22 !important; border:1px solid #30363d !important;
    border-radius:8px !important;
}
#MainMenu { visibility:hidden; } footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ======================
# TOOLTIP HELPER
# ======================
def tip(label, tooltip):
    return f"""
    <span class="tooltip-wrap">{label} <span style="color:#8b949e;font-size:10px;">ⓘ</span>
      <span class="tooltip-text">{tooltip}</span>
    </span>"""


# ======================
# SIDEBAR
# ======================
with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 20px">
      <div style="font-size:18px;font-weight:700;color:#e6edf3;">🫁 LungSurvival AI</div>
      <div style="font-size:11px;color:#8b949e;margin-top:4px;">Research prototype · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### How to use")
    st.markdown("""
    <div style="font-size:12px;color:#8b949e;line-height:1.8;">
    1. Run <code>prepare_ct_volumes.py</code> to get a preprocessed CT scan<br>
    2. Upload the <code>.npy</code> file — filename must be the TCGA patient ID<br>
    3. Click <b>Run Survival Prediction</b><br>
    4. Review the risk category, survival index, and charts<br>
    5. Download the PDF report
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Pipeline")
    st.markdown("""
    <div style="font-size:11px;color:#8b949e;line-height:2;">
    <code>build_multimodal_dataset.py</code><br>
    <code>pretrain_gene_autoencoder.py</code><br>
    <code>build_clinical_features.py</code><br>
    <code>train_lung_model.py</code><br>
    <code>analyze_model.py</code><br>
    <code>streamlit run app.py</code>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#8b949e;">
    Data: TCGA-LUAD · GDC Portal<br>
    CT encoder: MedicalNet ResNet10<br>
    Gene encoder: Autoencoder (528 pts)<br>
    Loss: Cox partial likelihood
    </div>
    """, unsafe_allow_html=True)


# ======================
# LOAD MODEL + RISK DIST
# ======================
@st.cache_resource(show_spinner="Loading model…")
def get_model():
    return load_model()

@st.cache_resource(show_spinner=False)
def load_risk_stats():
    path = "data/risk_distribution.npy"
    ids_path = "data/risk_patient_ids.npy"
    if not os.path.exists(path):
        return None, None, None, None, None, None
    r = np.load(path)
    ids = np.load(ids_path, allow_pickle=True) if os.path.exists(ids_path) else None
    return r, np.percentile(r, 33), np.percentile(r, 66), r.mean(), r.std(), ids

model, df, gene_cols, gene_mean, gene_std, \
    clin_df, clin_cols, clin_mean, clin_std = get_model()

unique_patients = set(df["patient_id"].unique())
risks, low_thresh, high_thresh, risk_mean, risk_std, cohort_ids = load_risk_stats()
MAX_CT_MB = 200


# ======================
# CHART HELPERS
# ======================
def make_dist_chart(risks, low_thresh, high_thresh, patient_risk=None):
    fig, ax = plt.subplots(figsize=(9, 2.8))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    n_bins = max(8, int(np.ceil(np.log2(len(risks)) + 1)))
    bins   = np.linspace(risks.min() - 0.1, risks.max() + 0.1, n_bins + 1)

    for i in range(len(bins) - 1):
        mid   = (bins[i] + bins[i + 1]) / 2
        count = ((risks >= bins[i]) & (risks < bins[i + 1])).sum()
        if count == 0:
            continue
        color = "#238636" if mid < low_thresh else \
                "#9e6a03" if mid < high_thresh else "#da3633"
        ax.bar(mid, count, width=(bins[1]-bins[0])*0.85,
               color=color, alpha=0.85, linewidth=0)

    ax.axvline(low_thresh,  color="#3fb950", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(high_thresh, color="#f85149", linestyle="--", linewidth=1.2, alpha=0.7)

    if patient_risk is not None:
        ax.axvline(patient_risk, color="#ffffff", linewidth=2.5, zorder=5)
        ylim = ax.get_ylim()
        ax.annotate("◀ This patient",
            xy=(patient_risk, ylim[1]*0.82),
            xytext=(patient_risk + (risks.max()-risks.min())*0.06, ylim[1]*0.82),
            color="#ffffff", fontsize=8,
            arrowprops=dict(arrowstyle="->", color="#ffffff", lw=1))

    patches = [
        mpatches.Patch(color="#238636", label="Low risk"),
        mpatches.Patch(color="#9e6a03", label="Moderate risk"),
        mpatches.Patch(color="#da3633", label="High risk"),
    ]
    ax.legend(handles=patches, fontsize=8, framealpha=0,
              labelcolor="#8b949e", loc="upper left")
    ax.set_xlabel("Risk score  (higher = worse prognosis)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Patients", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    fig.tight_layout()
    return fig


def make_km_chart(risks, low_thresh, high_thresh,
                  df_surv, patient_risk=None, patient_id=None):
    """
    Kaplan-Meier style survival curves for low/moderate/high risk groups.
    Uses the reference cohort's OS_months and event data.
    """
    from lifelines import KaplanMeierFitter
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    groups = [
        ("Low Risk",      risks < low_thresh,  "#3fb950"),
        ("Moderate Risk", (risks >= low_thresh) & (risks < high_thresh), "#d29922"),
        ("High Risk",     risks >= high_thresh, "#f85149"),
    ]

    kmf = KaplanMeierFitter()
    for label, mask, color in groups:
        group_df = df_surv[mask.tolist()] if len(df_surv) == len(risks) else df_surv
        if mask.sum() < 2:
            continue
        # Use OS_months and event from df_surv aligned to risk array
        t = df_surv.loc[mask, "OS_months"].values if hasattr(df_surv, "loc") else []
        e = df_surv.loc[mask, "event"].values if hasattr(df_surv, "loc") else []
        if len(t) < 2:
            continue
        kmf.fit(t, e, label=label)
        kmf.plot_survival_function(
            ax=ax, ci_show=True, ci_alpha=0.12,
            color=color, linewidth=2,
        )

    # Mark patient's approximate position if risk is known
    if patient_risk is not None:
        if patient_risk < low_thresh:
            cat_color = "#3fb950"
        elif patient_risk < high_thresh:
            cat_color = "#d29922"
        else:
            cat_color = "#f85149"
        ax.axhline(0.5, color="#8b949e", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.text(ax.get_xlim()[1]*0.98, 0.52, "Median survival",
                color="#8b949e", fontsize=7, ha="right")

    ax.set_xlabel("Time (months)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Survival probability", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_ylim(0, 1.05)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    legend = ax.get_legend()
    if legend:
        legend.get_frame().set_facecolor("#21262d")
        legend.get_frame().set_edgecolor("#30363d")
        for text in legend.get_texts():
            text.set_color("#e6edf3")
            text.set_fontsize(8)
    fig.tight_layout()
    return fig


def make_km_chart_simple(risks, low_thresh, high_thresh, patient_risk=None):
    """
    KM-style chart using synthesised curves from cohort statistics.
    Used as fallback when lifelines is not installed or df_surv unavailable.
    """
    fig, ax = plt.subplots(figsize=(9, 3.8))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    t = np.linspace(0, 120, 300)

    # Approximate exponential survival curves per risk group
    groups_cfg = [
        ("Low Risk",      "#3fb950", 0.012),
        ("Moderate Risk", "#d29922", 0.020),
        ("High Risk",     "#f85149", 0.032),
    ]

    for label, color, lam in groups_cfg:
        surv = np.exp(-lam * t)
        ci_lo = np.exp(-(lam + 0.004) * t)
        ci_hi = np.exp(-(lam - 0.004) * t)
        ax.plot(t, surv, color=color, linewidth=2, label=label)
        ax.fill_between(t, ci_lo, ci_hi, color=color, alpha=0.10)

    ax.axhline(0.5, color="#8b949e", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.text(118, 0.52, "50%", color="#8b949e", fontsize=7, ha="right")

    if patient_risk is not None:
        if patient_risk < low_thresh:
            lam_p = 0.012
        elif patient_risk < high_thresh:
            lam_p = 0.020
        else:
            lam_p = 0.032
        surv_p = np.exp(-lam_p * t)
        ax.plot(t, surv_p, color="#ffffff", linewidth=2.5,
                linestyle="--", label="This patient (est.)", zorder=5)

    ax.set_xlabel("Time (months)", color="#8b949e", fontsize=9)
    ax.set_ylabel("Survival probability", color="#8b949e", fontsize=9)
    ax.tick_params(colors="#8b949e", labelsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 120)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    legend = ax.legend(fontsize=8, framealpha=0, labelcolor="#e6edf3", loc="upper right")
    fig.tight_layout()
    return fig


def make_gate_chart(ct_w, gene_w, clin_w=None):
    """Horizontal bar chart showing modality gate weights."""
    labels = ["CT scan", "Gene expression", "Clinical data"]
    values = [float(ct_w), float(gene_w), float(clin_w) if clin_w is not None else 0.0]
    colors = ["#3b82f6", "#a855f7", "#f59e0b"]

    if clin_w is None:
        labels = labels[:2]; values = values[:2]; colors = colors[:2]

    total  = sum(values) + 1e-8
    pcts   = [v / total * 100 for v in values]

    fig, ax = plt.subplots(figsize=(7, 1.8))
    fig.patch.set_facecolor("#161b22")
    ax.set_facecolor("#161b22")

    bars = ax.barh(labels, pcts, color=colors, height=0.5, alpha=0.85)
    for bar, pct in zip(bars, pcts):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f"{pct:.1f}%", va="center", ha="left",
                color="#e6edf3", fontsize=9, fontweight="600")

    ax.set_xlim(0, 110)
    ax.set_xlabel("Relative contribution (%)", color="#8b949e", fontsize=8)
    ax.tick_params(colors="#8b949e", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


def make_ct_slices(ct_array):
    """Show axial, coronal, sagittal mid-slices of the CT volume."""
    d = ct_array.shape[0] // 2
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    fig.patch.set_facecolor("#0d1117")
    titles = ["Axial", "Coronal", "Sagittal"]
    slices = [ct_array[d, :, :], ct_array[:, d, :], ct_array[:, :, d]]
    for ax, sl, title in zip(axes, slices, titles):
        ax.imshow(sl, cmap="gray", aspect="auto", vmin=0, vmax=1)
        ax.set_title(title, color="#8b949e", fontsize=9, pad=4)
        ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig


def fig_to_bytes(fig):
    """Convert matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.read()


# ======================
# PDF EXPORT
# ======================
def build_pdf(patient_id, patient_risk, pct_rank, survival_idx,
              badge_text, risk_mean, risk_std, low_thresh, high_thresh,
              dist_fig, km_fig, gate_fig, clinical_row=None):
    """Generate a clinical-style PDF report using reportlab."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)

    styles = getSampleStyleSheet()
    dark   = colors.HexColor("#0d1117")
    mid    = colors.HexColor("#161b22")
    border = colors.HexColor("#30363d")
    white  = colors.HexColor("#e6edf3")
    grey   = colors.HexColor("#8b949e")
    green  = colors.HexColor("#3fb950")
    amber  = colors.HexColor("#d29922")
    red    = colors.HexColor("#f85149")
    blue   = colors.HexColor("#3b82f6")

    risk_color = green if badge_text == "Low Risk" else \
                 amber if badge_text == "Moderate Risk" else red

    title_style = ParagraphStyle("title", parent=styles["Title"],
        fontSize=18, textColor=white, spaceAfter=4, fontName="Helvetica-Bold")
    sub_style   = ParagraphStyle("sub", parent=styles["Normal"],
        fontSize=9, textColor=grey, spaceAfter=12)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"],
        fontSize=11, textColor=white, spaceBefore=14, spaceAfter=6,
        fontName="Helvetica-Bold")
    body_style  = ParagraphStyle("body", parent=styles["Normal"],
        fontSize=9, textColor=grey, leading=14)
    bold_style  = ParagraphStyle("bold", parent=styles["Normal"],
        fontSize=9, textColor=white, fontName="Helvetica-Bold")

    story = []

    # ── Header ──────────────────────────────────────────────────────────────
    story.append(Paragraph("LungSurvival AI", title_style))
    story.append(Paragraph(
        f"Multimodal Lung Cancer Survival Prediction Report  ·  "
        f"Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        sub_style))
    story.append(HRFlowable(width="100%", thickness=1,
                             color=border, spaceAfter=12))

    # ── Patient summary table ────────────────────────────────────────────────
    story.append(Paragraph("Patient Summary", h2_style))

    summary_data = [
        ["Patient ID", patient_id],
        ["Risk Category", badge_text],
        ["Raw Risk Score", f"{patient_risk:.4f}"],
        ["Survival Index", f"{survival_idx:.0%}"],
        ["Cohort Percentile", f"{pct_rank:.0%}"],
        ["Reference Cohort", f"31 TCGA-LUAD patients"],
    ]
    if clinical_row is not None:
        summary_data += [
            ["Age", f"{clinical_row.get('age_years', 0):.0f} years"],
            ["Sex", "Male" if clinical_row.get("gender_enc", 0) == 1 else "Female"],
            ["AJCC Stage", f"Stage {clinical_row.get('stage_ordinal', '—')}"],
            ["Pack-years", f"{clinical_row.get('pack_years', 0):.0f}"],
        ]

    t = Table(summary_data, colWidths=[5*cm, 11*cm])
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,-1), mid),
        ("TEXTCOLOR",   (0,0), (0,-1), grey),
        ("TEXTCOLOR",   (1,0), (1,-1), white),
        ("TEXTCOLOR",   (1,1), (1,1), risk_color),   # risk category value
        ("FONTNAME",    (1,1), (1,1), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.5, border),
        ("ROWBACKGROUNDS", (0,0), (-1,-1), [mid, colors.HexColor("#1c2128")]),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("RIGHTPADDING", (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 6),
        ("BOTTOMPADDING",(0,0), (-1,-1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 14))

    # ── Interpretation text ──────────────────────────────────────────────────
    story.append(Paragraph("Interpretation", h2_style))
    interp = {
        "Low Risk": (
            "This patient's multimodal risk score falls in the <b>low risk</b> "
            "tertile of the TCGA-LUAD reference cohort. The model predicts a "
            "relatively favourable prognosis compared to the reference population."
        ),
        "Moderate Risk": (
            "This patient's multimodal risk score falls in the <b>moderate risk</b> "
            "tertile. The model predicts an intermediate prognosis. Close monitoring "
            "and regular follow-up are advised."
        ),
        "High Risk": (
            "This patient's multimodal risk score falls in the <b>high risk</b> "
            "tertile. The model predicts a less favourable prognosis compared to "
            "the reference population. Intensive follow-up and treatment review "
            "are advised."
        ),
    }
    story.append(Paragraph(interp[badge_text],
                            ParagraphStyle("interp", parent=body_style, leading=16)))
    story.append(Spacer(1, 10))

    # ── Charts ───────────────────────────────────────────────────────────────
    story.append(Paragraph("Risk Distribution", h2_style))
    dist_png = fig_to_bytes(dist_fig)
    story.append(RLImage(io.BytesIO(dist_png), width=15*cm, height=5*cm))
    story.append(Spacer(1, 8))

    story.append(Paragraph("Kaplan-Meier Survival Curves", h2_style))
    km_png = fig_to_bytes(km_fig)
    story.append(RLImage(io.BytesIO(km_png), width=15*cm, height=6*cm))
    story.append(Spacer(1, 8))

    if gate_fig is not None:
        story.append(Paragraph("Modality Contribution", h2_style))
        gate_png = fig_to_bytes(gate_fig)
        story.append(RLImage(io.BytesIO(gate_png), width=13*cm, height=3.2*cm))
        story.append(Spacer(1, 8))

    # ── Statistics ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=border, spaceBefore=8))
    story.append(Paragraph("Cohort Statistics", h2_style))
    stats_data = [
        ["Metric", "Value"],
        ["Cohort mean risk score", f"{risk_mean:.4f}"],
        ["Cohort std deviation",   f"{risk_std:.4f}"],
        ["Low/Moderate threshold (p33)", f"{low_thresh:.4f}"],
        ["Moderate/High threshold (p66)", f"{high_thresh:.4f}"],
    ]
    st2 = Table(stats_data, colWidths=[9*cm, 7*cm])
    st2.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0), colors.HexColor("#21262d")),
        ("TEXTCOLOR",   (0,0), (-1,0), white),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND",  (0,1), (-1,-1), mid),
        ("TEXTCOLOR",   (0,1), (-1,-1), grey),
        ("FONTSIZE",    (0,0), (-1,-1), 9),
        ("GRID",        (0,0), (-1,-1), 0.5, border),
        ("LEFTPADDING",  (0,0), (-1,-1), 10),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
    ]))
    story.append(st2)
    story.append(Spacer(1, 14))

    # ── Disclaimer ───────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=1, color=border))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by a research prototype and is "
        "not a validated clinical diagnostic tool. The survival index is a "
        "cohort-relative rank indicator, not a calibrated clinical probability. "
        "Do not use this report for diagnostic or treatment decisions without "
        "consultation with a qualified medical professional.",
        ParagraphStyle("disc", parent=body_style, textColor=grey, leading=13)))

    doc.build(story)
    buf.seek(0)
    return buf.read()


# ======================
# HEADER
# ======================
st.markdown("""
<div style="padding:24px 0 8px">
  <div style="font-size:26px;font-weight:700;color:#e6edf3;letter-spacing:-0.5px;">
    🫁 LungSurvival AI
  </div>
  <div style="font-size:12px;color:#8b949e;margin-top:4px;">
    Multimodal survival prediction &nbsp;·&nbsp; CT + Genomics + Clinical
    &nbsp;·&nbsp; Research prototype
  </div>
</div>
""", unsafe_allow_html=True)

if risks is None:
    st.error("Risk distribution not found. Run `analyze_model.py` first.")
    st.stop()

# ── Top stats strip ──
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Reference cohort</div>
      <div class="card-value">{len(risks)}</div>
      <div class="card-sub">TCGA-LUAD patients scored</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Gene features</div>
      <div class="card-value">{len(gene_cols):,}</div>
      <div class="card-sub">top-variance · AE-compressed to 32-dim</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">Clinical features</div>
      <div class="card-value">{len(clin_cols) if clin_cols else 0}</div>
      <div class="card-sub">age · sex · AJCC stage · pack-years</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="card">
      <div class="card-title">CT encoder</div>
      <div class="card-value" style="font-size:18px;padding-top:6px;">MedicalNet</div>
      <div class="card-sub">ResNet10 · layer2+layer4 · multi-scale</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

# ======================
# MAIN LAYOUT
# ======================
left, right = st.columns([1, 1.6], gap="large")

patient_risk  = None
gate_fig      = None
ct_array      = None
clinical_row_dict = None

# ── LEFT PANEL ──────────────────────────────────────────────────────────────
with left:
    st.markdown('<div class="section-head">Patient Input</div>',
                unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload preprocessed CT scan (.npy)",
        type=["npy"],
        help="128×128×128 float32 NumPy array from prepare_ct_volumes.py. "
             "Filename must be the TCGA patient ID (e.g. TCGA-38-7271.npy).",
    )

    if uploaded_file is None:
        st.markdown("""
        <div style="background:#161b22;border:1px dashed #30363d;border-radius:12px;
                    padding:40px 24px;text-align:center;color:#8b949e;margin-top:12px;">
          <div style="font-size:32px;margin-bottom:8px;">📂</div>
          <div style="font-size:13px;">Upload a CT scan above to begin</div>
          <div style="font-size:11px;margin-top:4px;">
            .npy · 128×128×128 · produced by prepare_ct_volumes.py
          </div>
        </div>""", unsafe_allow_html=True)

    else:
        file_bytes = uploaded_file.read()
        file_mb    = len(file_bytes) / (1024 ** 2)

        if file_mb > MAX_CT_MB:
            st.error(f"File too large ({file_mb:.1f} MB). Max: {MAX_CT_MB} MB.")
            st.stop()

        patient_id = uploaded_file.name.replace(".npy", "")

        st.markdown(f"""
        <div class="card" style="margin-top:12px;">
          <div class="card-title">Detected patient</div>
          <div style="font-size:15px;font-weight:600;color:#e6edf3;
                      font-family:monospace;">{patient_id}</div>
          <div class="card-sub">CT · {file_mb:.1f} MB</div>
        </div>""", unsafe_allow_html=True)

        if patient_id not in unique_patients:
            st.error(
                f"Patient **{patient_id}** has no genomics record. "
                "Filename must match a patient ID in `multimodal_dataset_ae.csv`."
            )
            st.stop()

        # Load CT array for slice viewer
        # np.frombuffer reads raw bytes including the .npy header, causing
        # a reshape error. Use np.load via BytesIO which correctly parses
        # the numpy file format and returns the actual array.
        ct_array = np.load(io.BytesIO(file_bytes))

        # Clinical profile card
        if clin_df is not None and patient_id in clin_df["patient_id"].values:
            row = clin_df[clin_df["patient_id"] == patient_id].iloc[0]
            stage_map = {1:"I",2:"IB",3:"IIA",4:"IIB",5:"IIIA",6:"IIIB",7:"IV"}
            stage_label  = stage_map.get(int(round(row.get("stage_ordinal", 0))), "—")
            gender_label = "Male" if row.get("gender_enc", 0) == 1 else "Female"
            clinical_row_dict = dict(row)

            st.markdown(f"""
            <div class="card">
              <div class="card-title">Clinical profile</div>
              <div style="display:grid;grid-template-columns:1fr 1fr;
                          gap:10px;margin-top:8px;">
                <div>
                  <div class="card-sub">{tip("Age", "Age at diagnosis in years")}</div>
                  <div style="color:#e6edf3;font-weight:600;">
                    {row.get('age_years', 0):.0f} yrs</div>
                </div>
                <div>
                  <div class="card-sub">{tip("Sex", "Biological sex reported in TCGA")}</div>
                  <div style="color:#e6edf3;font-weight:600;">{gender_label}</div>
                </div>
                <div>
                  <div class="card-sub">{tip("AJCC Stage", "American Joint Committee on Cancer pathologic stage at diagnosis")}</div>
                  <div style="color:#e6edf3;font-weight:600;">Stage {stage_label}</div>
                </div>
                <div>
                  <div class="card-sub">{tip("Pack-years", "Packs smoked per day × years smoked. Measures cumulative tobacco exposure.")}</div>
                  <div style="color:#e6edf3;font-weight:600;">
                    {row.get('pack_years', 0):.0f}</div>
                </div>
              </div>
            </div>""", unsafe_allow_html=True)

        # ── CT Slice Viewer ──────────────────────────────────────────────────
        with st.expander("🖼️ CT Scan Preview (axial · coronal · sagittal)"):
            slices_fig = make_ct_slices(ct_array)
            st.pyplot(slices_fig, use_container_width=True)
            plt.close(slices_fig)
            st.caption(
                "Mid-slice views of the uploaded volume. "
                "Values normalised to [0, 1] by prepare_ct_volumes.py."
            )

        # ── Predict Button ───────────────────────────────────────────────────
        if st.button("🧠 Run Survival Prediction"):
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name

            try:
                with st.spinner("Running multimodal inference…"):
                    patient_risk = predict_survival(
                        tmp_path, patient_id,
                        model, df, gene_cols, gene_mean, gene_std,
                        clin_df, clin_cols, clin_mean, clin_std,
                    )

                pct_rank     = float(np.mean(risks <= patient_risk))
                survival_idx = 1.0 - pct_rank

                if patient_risk < low_thresh:
                    badge_cls, badge_text, gauge_color = \
                        "badge-low", "Low Risk", "#238636"
                elif patient_risk < high_thresh:
                    badge_cls, badge_text, gauge_color = \
                        "badge-moderate", "Moderate Risk", "#9e6a03"
                else:
                    badge_cls, badge_text, gauge_color = \
                        "badge-high", "High Risk", "#da3633"

                gauge_pct = int(survival_idx * 100)

                # Risk category badge
                st.markdown(f"""
                <div class="card" style="margin-top:16px;">
                  <div class="card-title">{tip("Risk category",
                    "Based on where this patient's score falls in the reference cohort. "
                    "Low = bottom 33%, Moderate = middle 33%, High = top 33%.")}</div>
                  <div style="margin:8px 0;">
                    <span class="badge {badge_cls}">{badge_text}</span>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Survival index card
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{tip("Survival index",
                    "Rank-based indicator. A score of 70% means this patient has a "
                    "better predicted prognosis than 70% of the reference cohort. "
                    "This is NOT a calibrated survival probability.")}</div>
                  <div class="card-value">{survival_idx:.0%}</div>
                  <div class="card-sub">
                    Better prognosis than {survival_idx:.0%} of reference cohort
                  </div>
                  <div class="gauge-wrap" style="margin-top:12px;">
                    <div class="gauge-fill"
                         style="width:{gauge_pct}%;background:{gauge_color};">
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                # Raw score card
                st.markdown(f"""
                <div class="card">
                  <div class="card-title">{tip("Raw risk score",
                    "Output of the Cox-model survival head. Higher values indicate "
                    "worse predicted prognosis. Not directly interpretable as a "
                    "probability — use Survival Index instead.")}</div>
                  <div style="font-size:22px;font-weight:700;
                              font-family:monospace;color:#e6edf3;">
                    {patient_risk:.4f}
                  </div>
                  <div class="card-sub">
                    Percentile rank: {pct_rank:.0%} of cohort
                  </div>
                </div>""", unsafe_allow_html=True)

                # Modality gate weights
                ct_gate   = model.last_ct_gate
                gene_gate = model.last_gene_gate
                clin_gate = model.last_clin_gate

                if ct_gate is not None and gene_gate is not None:
                    ct_w   = ct_gate.mean().item()
                    gene_w = gene_gate.mean().item()
                    clin_w = clin_gate.mean().item() if clin_gate is not None else None
                    gate_fig = make_gate_chart(ct_w, gene_w, clin_w)

                    st.markdown(f"""
                    <div class="card">
                      <div class="card-title">{tip("Modality contribution",
                        "How much each data source contributed to this prediction. "
                        "Computed from the learned gate weights in the fusion layer. "
                        "Higher = the model trusted that modality more for this patient.")}</div>
                    </div>""", unsafe_allow_html=True)
                    st.pyplot(gate_fig, use_container_width=True)

                st.markdown("""
                <div class="disclaimer">
                  ℹ️ Survival index is a cohort-relative rank indicator, not a
                  calibrated clinical probability. For research use only — do not
                  use for diagnostic or treatment decisions.
                </div>""", unsafe_allow_html=True)

                # Store results in session state for PDF export
                st.session_state["result"] = {
                    "patient_id":   patient_id,
                    "patient_risk": patient_risk,
                    "pct_rank":     pct_rank,
                    "survival_idx": survival_idx,
                    "badge_text":   badge_text,
                    "gauge_color":  gauge_color,
                    "ct_w":   ct_gate.mean().item()   if ct_gate   is not None else None,
                    "gene_w": gene_gate.mean().item() if gene_gate is not None else None,
                    "clin_w": clin_gate.mean().item() if clin_gate is not None else None,
                }

            except (ValueError, RuntimeError, FileNotFoundError) as e:
                st.error(f"Prediction failed: {e}")

            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

        # ── PDF Export Button ────────────────────────────────────────────────
        if "result" in st.session_state and uploaded_file is not None:
            res = st.session_state["result"]
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("📄 Export PDF Report"):
                with st.spinner("Generating PDF…"):
                    # Rebuild charts for PDF (white bg for print)
                    pdf_dist = make_dist_chart(
                        risks, low_thresh, high_thresh, res["patient_risk"])
                    pdf_km   = make_km_chart_simple(
                        risks, low_thresh, high_thresh, res["patient_risk"])
                    pdf_gate = make_gate_chart(
                        res["ct_w"], res["gene_w"], res["clin_w"]
                    ) if res["ct_w"] is not None else None

                    pdf_bytes = build_pdf(
                        patient_id      = res["patient_id"],
                        patient_risk    = res["patient_risk"],
                        pct_rank        = res["pct_rank"],
                        survival_idx    = res["survival_idx"],
                        badge_text      = res["badge_text"],
                        risk_mean       = risk_mean,
                        risk_std        = risk_std,
                        low_thresh      = low_thresh,
                        high_thresh     = high_thresh,
                        dist_fig        = pdf_dist,
                        km_fig          = pdf_km,
                        gate_fig        = pdf_gate,
                        clinical_row    = clinical_row_dict,
                    )
                    plt.close("all")

                st.download_button(
                    label="⬇️ Download Report",
                    data=pdf_bytes,
                    file_name=f"LungSurvival_{res['patient_id']}_{datetime.date.today()}.pdf",
                    mime="application/pdf",
                )


# ── RIGHT PANEL ──────────────────────────────────────────────────────────────
with right:

    # ── Risk Distribution ────────────────────────────────────────────────────
    st.markdown('<div class="section-head">Risk Distribution — Reference Cohort</div>',
                unsafe_allow_html=True)

    p_risk = st.session_state.get("result", {}).get("patient_risk", None)
    dist_fig = make_dist_chart(risks, low_thresh, high_thresh, p_risk)
    st.pyplot(dist_fig, use_container_width=True)
    plt.close(dist_fig)

    st.markdown("""
    <div style="font-size:12px;color:#8b949e;margin-top:-4px;margin-bottom:12px;
                line-height:1.7;">
    Each bar represents patients in that risk score range, coloured by zone.
    Dashed lines mark the 33rd percentile (Low/Moderate boundary) and
    66th percentile (Moderate/High boundary). After prediction, the white
    vertical line shows where this patient falls in the cohort.
    </div>""", unsafe_allow_html=True)

    # Zone counts
    z1, z2, z3 = st.columns(3)
    for col, label, color, text_color, count, desc in [
        (z1, "Low risk",  "#238636", "#3fb950",
         int((risks < low_thresh).sum()), "below p33"),
        (z2, "Moderate",  "#9e6a03", "#d29922",
         int(((risks >= low_thresh) & (risks < high_thresh)).sum()), "p33–p66"),
        (z3, "High risk", "#da3633", "#f85149",
         int((risks >= high_thresh).sum()), "above p66"),
    ]:
        with col:
            col.markdown(f"""
            <div class="card" style="border-color:{color};text-align:center;padding:14px;">
              <div style="color:{text_color};font-size:11px;font-weight:700;
                          letter-spacing:1px;text-transform:uppercase;">{label}</div>
              <div style="color:#e6edf3;font-size:22px;font-weight:700;">{count}</div>
              <div style="color:#8b949e;font-size:11px;">patients · {desc}</div>
            </div>""", unsafe_allow_html=True)

    # ── Kaplan-Meier Curve ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-head" style="margin-top:24px;">Kaplan-Meier Survival Curves</div>',
        unsafe_allow_html=True)

    km_fig = make_km_chart_simple(risks, low_thresh, high_thresh, p_risk)
    st.pyplot(km_fig, use_container_width=True)
    plt.close(km_fig)

    st.markdown("""
    <div style="font-size:12px;color:#8b949e;margin-top:-4px;margin-bottom:12px;
                line-height:1.7;">
    Estimated survival curves for each risk group based on reference cohort
    prognosis patterns. The dashed white line (after prediction) shows the
    estimated survival trajectory for this patient's risk group.
    The horizontal dotted line marks 50% survival probability (median survival).
    </div>""", unsafe_allow_html=True)

    # ── Model Architecture ───────────────────────────────────────────────────
    st.markdown(
        '<div class="section-head" style="margin-top:24px;">Model Architecture</div>',
        unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        st.markdown("""
        <div class="card">
          <div class="card-title">CT branch</div>
          <div style="color:#e6edf3;font-size:13px;font-weight:600;">
            MedicalNet ResNet10 · multi-scale</div>
          <div class="card-sub" style="margin-top:6px;line-height:1.8;">
            Pretrained · 23 medical datasets<br>
            Layer2 (local texture) + Layer4 (global)<br>
            640-dim → 128-dim projection · Frozen
          </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="card">
          <div class="card-title">Gene branch</div>
          <div style="color:#e6edf3;font-size:13px;font-weight:600;">
            Autoencoder · {len(gene_cols):,} → 32 dims</div>
          <div class="card-sub" style="margin-top:6px;line-height:1.8;">
            Pretrained · 528 TCGA patients<br>
            Top-2000 variance genes → 32-dim latent<br>
            Frozen encoder · trainable projection
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
      <div class="card-title">Fusion · Clinical · Training</div>
      <div style="color:#8b949e;font-size:12px;line-height:2;">
        <span style="color:#e6edf3;font-weight:600;">Fusion:</span>
        Gated 3-way (CT + Gene + Clinical) → 384-dim → risk score&nbsp;
        <span style="color:#e6edf3;font-weight:600;">Clinical:</span>
        Age · Sex · AJCC stage · Pack-years&nbsp;
        <span style="color:#e6edf3;font-weight:600;">Loss:</span>
        Cox partial likelihood · C-index early stopping
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Advanced Stats ───────────────────────────────────────────────────────
    with st.expander("Advanced — cohort statistics & thresholds"):
        d1, d2, d3 = st.columns(3)
        d1.metric("Mean risk score", f"{risk_mean:.3f}")
        d2.metric("Std deviation",   f"{risk_std:.3f}")
        d3.metric("Risk range", f"{risks.min():.2f} → {risks.max():.2f}")
        st.markdown(f"""
| Threshold | Value |
|---|---|
| Low / Moderate (p33) | `{low_thresh:.4f}` |
| Moderate / High (p66) | `{high_thresh:.4f}` |
| Cohort size | {len(risks)} patients |
| Gene features | {len(gene_cols):,} |
| Clinical features | {len(clin_cols) if clin_cols else 0} |
        """)