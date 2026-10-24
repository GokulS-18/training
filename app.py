import os
import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import ml

st.set_page_config(page_title="MPG Predictor", page_icon="🚗", layout="wide")

# ---------- global styling + animated header ----------
st.markdown("""
<style>
.stApp { background: linear-gradient(160deg,#0f172a 0%,#1e293b 60%,#0f172a 100%); }
h1,h2,h3,p,label,span,div { color:#e2e8f0; }
.hero { padding:18px 24px 0; }
.hero h1 { font-size:2.4rem; margin:0; background:linear-gradient(90deg,#38bdf8,#a78bfa,#f472b6);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#94a3b8; margin:4px 0 0; }
.road { position:relative; height:70px; margin:10px 0 18px; border-radius:12px; overflow:hidden; background:#1f2937; }
.road::after { content:""; position:absolute; left:0; right:0; top:50px; height:4px;
  background:repeating-linear-gradient(90deg,#fbbf24 0 30px,transparent 30px 60px);
  background-size:60px 4px; animation:dash 0.6s linear infinite; }
.car { position:absolute; bottom:18px; left:-120px; font-size:2.4rem; animation:drive 7s linear infinite; }
.car span { display:inline-block; animation:bob .35s ease-in-out infinite alternate; }
@keyframes dash { to { background-position:-60px 0; } }
@keyframes drive { to { left:105%; } }
@keyframes bob { to { transform:translateY(-3px); } }
div[data-testid="stMetric"] { background:#1e293b; border:1px solid #334155; padding:12px; border-radius:12px; }
.stButton>button { background:linear-gradient(90deg,#0ea5e9,#8b5cf6); color:white; border:0;
  border-radius:10px; padding:.6rem 1.4rem; font-weight:600; transition:transform .15s; }
.stButton>button:hover { transform:translateY(-2px) scale(1.03); }
</style>
<div class="hero"><h1>🏃🏻‍➡️ Fuel Efficiency (MPG) Predictor</h1>
<p>Upload data → train &amp; compare ML models → predict a vehicle's mileage.</p></div>
<div class="road"><div class="car"><span>🏃🏻‍➡️</span></div></div>
""", unsafe_allow_html=True)


def result_widget(mpg: float) -> str:
    """Animated gauge + count-up + driving car (self-contained HTML/JS)."""
    mpg = max(0.0, mpg)
    kmpl = mpg * 0.425144
    l100 = 235.215 / mpg if mpg > 0 else 0
    label, color = (("Excellent efficiency", "#22c55e") if mpg >= 30 else
                    ("Average efficiency", "#facc15") if mpg >= 20 else ("Thirsty engine", "#ef4444"))
    return f"""
<div style="font-family:system-ui,sans-serif;color:#e2e8f0;text-align:center;background:#1e293b;
 border:1px solid #334155;border-radius:16px;padding:16px;">
 <svg viewBox="0 0 300 175" width="340">
  <path d="M30 150 A120 120 0 0 1 270 150" fill="none" stroke="#334155" stroke-width="18" stroke-linecap="round"/>
  <path id="arc" d="M30 150 A120 120 0 0 1 270 150" fill="none" stroke="{color}" stroke-width="18"
        stroke-linecap="round" stroke-dasharray="377" stroke-dashoffset="377"/>
  <g id="needle"><line x1="150" y1="150" x2="150" y2="48" stroke="#f8fafc" stroke-width="4" stroke-linecap="round"/></g>
  <circle cx="150" cy="150" r="9" fill="#f8fafc"/>
  <text x="22" y="170" fill="#94a3b8" font-size="11">0</text>
  <text x="262" y="170" fill="#94a3b8" font-size="11">50+</text>
 </svg>
 <div style="font-size:3rem;font-weight:800;color:{color};margin-top:-8px"><span id="num">0.0</span>
   <span style="font-size:1.1rem;color:#94a3b8">MPG</span></div>
 <div style="font-weight:600;margin:2px 0 8px">{label}</div>
 <div style="color:#94a3b8;font-size:.9rem">≈ {kmpl:.1f} km/L &nbsp;•&nbsp; {l100:.1f} L/100 km</div>
 <div style="position:relative;height:46px;margin-top:10px;overflow:hidden;border-radius:10px;background:#111827">
  <div id="rd" style="position:absolute;left:0;right:0;bottom:8px;height:3px;
   background:repeating-linear-gradient(90deg,#fbbf24 0 24px,transparent 24px 48px);"></div>
  <div id="car" style="position:absolute;bottom:10px;left:0;font-size:1.8rem;">🏃🏻‍➡️</div>
 </div>
</div>
<script>
const target={mpg:.2f}, frac=Math.min(target/50,1), T=1400, t0=performance.now();
const ease=t=>1-Math.pow(1-t,3);
function step(now){{
  const p=ease(Math.min((now-t0)/T,1));
  document.getElementById('num').textContent=(target*p).toFixed(1);
  document.getElementById('needle').setAttribute('transform','rotate('+(-90+180*frac*p)+' 150 150)');
  document.getElementById('arc').setAttribute('stroke-dashoffset',377*(1-frac*p));
  if(p<1) requestAnimationFrame(step);
}}
document.getElementById('needle').setAttribute('transform','rotate(-90 150 150)');
requestAnimationFrame(step);
// car speed reflects efficiency: better MPG = faster cruise
const car=document.getElementById('car'); let x=-40, v=1.5+frac*4;
(function move(){{ x+=v; if(x>car.parentElement.clientWidth) x=-40; car.style.left=x+'px'; requestAnimationFrame(move); }})();
</script>
"""


# ---------- sidebar: data upload ----------
with st.sidebar:
    st.header("📁 Dataset")
    up = st.file_uploader("Upload auto-mpg CSV", type=["csv"])
    st.caption("Needs: mpg, cylinders, displacement, horsepower, weight, acceleration, "
               "model year, origin (+ optional car name). '?' values are handled automatically.")
    train_btn = st.button("⚙️ Train & select best model", disabled=up is None)

if train_btn and up is not None:
    try:
        with st.spinner("Cleaning data, cross-validating 4 algorithms…"):
            df = ml.load_and_clean(up)
            ml.train_and_select(df)
            st.session_state["df"] = df
        st.sidebar.success("Model trained and saved!")
    except Exception as e:
        st.sidebar.error(f"Could not process file: {e}")

bundle = joblib.load(ml.MODEL_PATH) if os.path.exists(ml.MODEL_PATH) else None

tab_pred, tab_models, tab_data = st.tabs(["🔮 Predict", "📊 Model comparison", "🗂️ Data"])

# ---------- predict ----------
with tab_pred:
    if bundle is None:
        st.info("👈 Upload your dataset in the sidebar and click **Train & select best model** to begin.")
    else:
        r = bundle["ranges"]
        left, right = st.columns([1.2, 1])
        with left:
            st.subheader("Vehicle specifications")
            c1, c2 = st.columns(2)
            cyl = c1.selectbox("Cylinders", [3, 4, 5, 6, 8], index=1)
            origin = c2.selectbox("Origin", ["USA", "Europe", "Japan"])
            disp = st.slider("Displacement (cu in)", r["displacement"][0], r["displacement"][1], r["displacement"][2])
            known_hp = st.checkbox("I know the horsepower", value=True)
            hp = st.slider("Horsepower", r["horsepower"][0], r["horsepower"][1], r["horsepower"][2],
                           disabled=not known_hp)
            weight = st.slider("Weight (lbs)", r["weight"][0], r["weight"][1], r["weight"][2])
            acc = st.slider("Acceleration (0-60, sec)", r["acceleration"][0], r["acceleration"][1], r["acceleration"][2])
            year = st.slider("Model year (19xx)", int(r["model_year"][0]), int(r["model_year"][1]),
                             int(r["model_year"][2]))
            brand = st.selectbox("Brand (optional)", ["(unknown)"] + bundle["brands"])
            go = st.button("🚀 Predict MPG")
        with right:
            st.subheader("Prediction")
            if go:
                specs = dict(cylinders=cyl, displacement=disp, horsepower=hp if known_hp else None,
                             weight=weight, acceleration=acc, model_year=year, origin=origin,
                             brand=None if brand == "(unknown)" else brand)
                mpg = ml.predict_mpg(bundle["pipeline"], specs)
                components.html(result_widget(mpg), height=420)
                st.balloons() if mpg >= 30 else None
            else:
                st.caption("Set the specs and press **Predict MPG**.")
            st.caption(f"Deployed model: **{bundle['name']}**")

# ---------- model comparison ----------
with tab_models:
    if bundle is None:
        st.info("Train a model first.")
    else:
        res = bundle["results"]
        st.success(f"🏆 Best model (lowest CV RMSE): **{bundle['name']}**")
        st.dataframe(res.style.format({c: "{:.3f}" for c in res.columns if c != "Model"}),
                     use_container_width=True, hide_index=True)
        st.bar_chart(res.set_index("Model")[["Test R²", "CV R²"]])
        st.bar_chart(res.set_index("Model")[["Test RMSE", "Test MAE"]])

# ---------- data ----------
with tab_data:
    if "df" in st.session_state:
        d = st.session_state["df"]
        st.write(f"{len(d)} rows • missing values handled in the pipeline (median / most frequent).")
        st.dataframe(d.head(50), use_container_width=True)
        st.dataframe(d.isna().sum().rename("missing values").to_frame().T)
    else:
        st.info("Upload and train to preview your data here.")
