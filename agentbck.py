import os
import re
import uuid
import numpy as np
import pandas as pd
import shap
from fpdf import FPDF
from agent_platform.agent_sdk import Agent
from model_utils import ensure_model, FEATURES

# Unicode-safe system font (install: sudo apt-get install -y fonts-dejavu-core)
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

agent = Agent(name="credit_appraisal", root=os.path.dirname(__file__))


def _soften_tokens(s: str, maxlen: int = 60) -> str:
    """Insert spaces into very long tokens so MultiCell can wrap."""
    return re.sub(rf"(\S{{{maxlen}}})", r"\1 ", s)


def _sanitize_text(s: str, keep_unicode: bool) -> str:
    """
    Make text safe for PDF:
    - replace common Unicode punctuation with ASCII equivalents,
    - insert soft breaks into very long tokens,
    - if not using a Unicode font, strip to Latin-1.
    """
    if s is None:
        return ""
    repl = {
        "•": "- ",
        "–": "-",
        "—": "-",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "\u00A0": " ",  # non-breaking space
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    s = _soften_tokens(s, maxlen=60)
    if keep_unicode:
        return s
    return s.encode("latin-1", "ignore").decode("latin-1")


def _safe_multicell(pdf: FPDF, text: str, line_h: float = 6):
    """Write using epw; fallback by shrinking font if needed."""
    try:
        epw = pdf.w - pdf.l_margin - pdf.r_margin
    except Exception:
        epw = 190
    try:
        pdf.multi_cell(w=epw, h=line_h, txt=text, new_x="LMARGIN", new_y="NEXT")
        return
    except Exception:
        pass
    for size in (pdf.font_size_pt - 1, pdf.font_size_pt - 2, 9, 8):
        if size < 7:
            break
        try:
            pdf.set_font(pdf.font_family, pdf.font_style, size)
            pdf.multi_cell(w=epw, h=line_h - 1, txt=text, new_x="LMARGIN", new_y="NEXT")
            return
        except Exception:
            continue
    safer = text.encode("ascii", "ignore").decode("ascii")
    pdf.set_font(pdf.font_family, pdf.font_style, 9)
    pdf.multi_cell(w=epw, h=5, txt=safer, new_x="LMARGIN", new_y="NEXT")


def make_pdf(report_path: str, items: list[dict], summary: dict, narrative: str | None):
    pdf = FPDF()
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    use_dejavu = os.path.exists(FONT_PATH)
    if use_dejavu:
        pdf.add_font("DejaVu", "", FONT_PATH, uni=True)
        pdf.add_font("DejaVu", "B", FONT_PATH, uni=True)
        base = "DejaVu"
    else:
        base = "Arial"

    pdf.set_font(base, "B", 16)
    _safe_multicell(pdf, "Credit Appraisal Report")

    pdf.set_font(base, "", 11)
    _safe_multicell(pdf, f"Total applications: {summary.get('count', 0)}")
    _safe_multicell(pdf, f"Approved: {summary.get('approved', 0)} | Denied: {summary.get('denied', 0)}")

    if narrative:
        pdf.ln(2)
        pdf.set_font(base, "B", 11)
        _safe_multicell(pdf, "Narrative:")
        pdf.set_font(base, "", 10)
        safe_narr = _sanitize_text(narrative, keep_unicode=use_dejavu)
        _safe_multicell(pdf, safe_narr)

    pdf.ln(2)
    pdf.set_font(base, "B", 11)
    _safe_multicell(pdf, "Top Explanations:")
    pdf.set_font(base, "", 10)

    for it in items[:10]:
        line = (
            f"{it['application_id']}: decision={it['decision']}, "
            f"score={it['score']:.3f}, reason={it['explanation']}"
        )
        safe_line = _sanitize_text(line, keep_unicode=use_dejavu)
        _safe_multicell(pdf, safe_line)

    pdf.output(report_path)


@agent.runner
def run(inputs: dict, ctx: dict):
    csv_path = inputs["applications_csv"]
    df = pd.read_csv(csv_path)

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    model = ensure_model(df)
    X = df[FEATURES]
    proba = model.predict_proba(X)[:, 1]
    decision = np.where(proba >= 0.5, "approve", "deny")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    items: list[dict] = []
    for i in range(len(df)):
        sv = shap_values[i]
        idx = int(np.argmax(np.abs(sv)))
        feature = FEATURES[idx]
        direction = "increased" if sv[idx] > 0 else "decreased"
        explanation = f"{feature} {direction} approval likelihood most"
        items.append(
            {
                "application_id": str(df.iloc[i].get("application_id", f"row_{i}")),
                "score": float(proba[i]),
                "decision": decision[i],
                "top_feature": feature,
                "shap": float(sv[idx]),
                "explanation": explanation,
            }
        )

    approved = int((decision == "approve").sum())
    summary = {
        "count": len(df),
        "approved": approved,
        "denied": int(len(df) - approved),
    }

    narrative = ctx.get("narrative") if isinstance(ctx, dict) else None

    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "services", "api", ".runs")
    os.makedirs(out_dir, exist_ok=True)
    uid = uuid.uuid4().hex

    # CSV artifacts for reuse
    scores_csv = os.path.join(out_dir, f"scores_{uid}.csv")
    explanations_csv = os.path.join(out_dir, f"explanations_{uid}.csv")

    pd.DataFrame(
        [{"application_id": it["application_id"], "score": it["score"], "decision": it["decision"]} for it in items]
    ).to_csv(scores_csv, index=False)

    pd.DataFrame(items).to_csv(explanations_csv, index=False)

    # Optional PDF (keep generating if you want)
    report_path = os.path.join(out_dir, f"credit_report_{uid}.pdf")
    try:
        make_pdf(report_path, items, summary, narrative)
        pdf_ok = True
    except Exception as _:
        # If PDF fails, we still deliver CSVs
        pdf_ok = False
        report_path = None

    artifacts = {
        "scores_csv": scores_csv,
        "explanations_csv": explanations_csv,
    }
    if pdf_ok and report_path:
        artifacts["explanation_pdf"] = report_path

    return {
        "scores": [{"application_id": it["application_id"], "score": it["score"], "decision": it["decision"]} for it in items],
        "explanations": items,
        "summary": summary,
        "artifacts": artifacts,
    }
