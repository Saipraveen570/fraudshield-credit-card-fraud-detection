# notebooks/shap_report.py
import base64, os, json
from pathlib import Path
from datetime import datetime

OUT_DIR = Path("reports/shap")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def b64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def main():
    beeswarm = OUT_DIR / "shap_summary_beeswarm.png"
    barplot  = OUT_DIR / "shap_summary_bar.png"
    waterfall = OUT_DIR / "shap_waterfall_top_case.png"
    top_json = OUT_DIR / "top_features.json"

    # Helpful message if files are missing
    missing = [str(p) for p in [beeswarm, barplot] if not p.exists()]
    if missing:
        raise SystemExit(
            "Missing SHAP assets:\n- " + "\n- ".join(missing) +
            "\n\nRun:  .\\.venv\\Scripts\\activate.bat  &&  python notebooks\\shap_explain.py"
        )

    top_feats_html = ""
    if top_json.exists():
        with open(top_json, "r") as f:
            arr = json.load(f)
        rows = "\n".join(
            f"<tr><td>{i+1}</td><td>{x['feature']}</td><td>{x['mean_abs_shap']:.6f}</td></tr>"
            for i, x in enumerate(arr[:20])
        )
        top_feats_html = f"""
        <h2 id="top20">Top 20 Features (mean |SHAP|)</h2>
        <table>
          <thead><tr><th>#</th><th>Feature</th><th>mean |SHAP|</th></tr></thead>
          <tbody>{rows}</tbody>
        </table>
        """

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>FraudShield — SHAP Report</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>
  :root {{
    --bg:#0f1117; --fg:#e6edf3; --muted:#9aa4af; --card:#151a23; --accent:#6aa0ff;
    --border:#222834; --table:#1b2230;
  }}
  body {{ margin:0; font-family: ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto; background:var(--bg); color:var(--fg); }}
  .wrap {{ max-width:1100px; margin:40px auto; padding:0 20px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:16px; padding:24px; margin-bottom:20px; box-shadow:0 1px 10px rgba(0,0,0,0.25); }}
  h1,h2 {{ margin:0 0 12px; }}
  p.muted {{ color:var(--muted); margin:4px 0 0; }}
  .row {{ display:flex; gap:20px; flex-wrap:wrap; }}
  .imgbox {{ flex:1 1 520px; }}
  .imgbox img {{ width:100%; border-radius:12px; border:1px solid var(--border); }}
  code.badge {{ background: var(--table); padding:2px 8px; border-radius:8px; border:1px solid var(--border); color:var(--muted); }}
  table {{ width:100%; border-collapse:collapse; background:var(--table); border:1px solid var(--border); border-radius:12px; overflow:hidden; }}
  th, td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }}
  thead th {{ background:#0d131d; color:#c8d1dc; font-weight:600; }}
  footer {{ color:var(--muted); text-align:center; margin:20px 0 40px; }}
  a {{ color: var(--accent); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>FraudShield — SHAP Explainability Report</h1>
      <p class="muted">Generated: {datetime.utcnow().isoformat()} UTC</p>
      <p class="muted">Artifacts from <code class="badge">reports/shap/</code></p>
    </div>

    <div class="card">
      <h2 id="global">Global Importance (Beeswarm)</h2>
      <div class="imgbox">
        <img alt="SHAP beeswarm" src="data:image/png;base64,{b64(beeswarm)}" />
      </div>
    </div>

    <div class="card">
      <h2 id="bars">Global Importance (mean |SHAP|)</h2>
      <div class="imgbox">
        <img alt="SHAP bar" src="data:image/png;base64,{b64(barplot)}" />
      </div>
    </div>

    {"".join([f'<div class="card"><h2 id="waterfall">Top Case Waterfall</h2><div class="imgbox"><img alt="Waterfall" src="data:image/png;base64,' + b64(waterfall) + '" /></div></div>' if waterfall.exists() else ""])}

    <div class="card">
      {top_feats_html}
      <p class="muted">Use this to describe which features most influenced fraud predictions.</p>
    </div>

    <footer>
      &copy; FraudShield — SHAP report · <a href="./">Back to project</a>
    </footer>
  </div>
</body>
</html>
"""
    out_html = OUT_DIR / "shap_report.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Wrote {out_html.resolve()}")

if __name__ == "__main__":
    main()