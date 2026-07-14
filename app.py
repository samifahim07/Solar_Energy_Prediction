# ================================================================
#  SolarIQ — Solar Energy Prediction
#  Single file: Flask backend + HTML frontend embedded
#
#  HOW TO RUN:
#    Step 1: pip install flask flask-cors numpy pandas scikit-learn
#    Step 2: Make sure random_forest_model.pkl is in same folder
#            (run your Jupyter notebook to generate it first)
#    Step 3: python app.py
#    Step 4: Open http://localhost:5000 in browser
# ================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import pandas as pd

# ── Create Flask app ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ── Load the saved Random Forest model ──────────────────────────
# This file is created when you run the last cell of your notebook
with open("random_forest_model.pkl", "rb") as file:
    rf = pickle.load(file)

# ── These are the EXACT columns used in X inside the notebook ───
# X = new_df[["GHI","temp","pressure","humidity","wind_speed",
#              "rain_1h","clouds_all","isSun","sunlightTime","hour","month"]]
FEATURE_COLUMNS = [
    "GHI",
    "temp",
    "pressure",
    "humidity",
    "wind_speed",
    "rain_1h",
    "clouds_all",
    "isSun",
    "sunlightTime",
    "hour",
    "month"
]


# ── Route: Serve the HTML dashboard ─────────────────────────────
@app.route("/")
def index():
    return HTML_PAGE   # HTML is embedded at the bottom of this file


# ── Route: Health check ──────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": "Random Forest",
        "features": FEATURE_COLUMNS
    })


# ── Route: Predict energy output ────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get JSON sent from the browser form
        data = request.get_json(force=True)

        # 2. Check all required fields are present
        missing = [f for f in FEATURE_COLUMNS if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # 3. Validate basic ranges
        if not (0 <= int(data.get("hour", 0)) <= 23):
            return jsonify({"error": "hour must be between 0 and 23"}), 400
        if not (1 <= int(data.get("month", 1)) <= 12):
            return jsonify({"error": "month must be between 1 and 12"}), 400
        if not (0 <= float(data.get("clouds_all", 0)) <= 100):
            return jsonify({"error": "clouds_all must be between 0 and 100"}), 400

        # 4. Build the feature DataFrame (same order as notebook)
        row = {col: float(data[col]) for col in FEATURE_COLUMNS}
        X_input = pd.DataFrame([row])

        # 5. Run prediction
        prediction = float(rf.predict(X_input)[0])
        prediction = max(0.0, round(prediction, 2))   # energy can't be negative

        # 6. Assign a human-readable level label
        if prediction == 0:
            level = "None"
        elif prediction < 500:
            level = "Low"
        elif prediction < 2000:
            level = "Medium"
        elif prediction < 3500:
            level = "High"
        else:
            level = "Peak"

        # 7. Return result as JSON
        return jsonify({
            "predicted_energy_Wh": prediction,
            "unit": "Watt-hours",
            "level": level,
            "model": "Random Forest",
            "pct_of_max": round(prediction / 5020 * 100, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── HTML page (no separate index.html file needed) ───────────────
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SolarIQ — Energy Prediction</title>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Space+Mono:wght@400;700&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg:       #0b0f1a;
      --surface:  #111827;
      --surface2: #1a2436;
      --border:   #1f2f45;
      --gold:     #f5c842;
      --gold-dim: #c49a1a;
      --sky:      #38bdf8;
      --sky-dim:  #1a7aa8;
      --muted:    #64748b;
      --text:     #e2e8f0;
      --text-dim: #94a3b8;
      --success:  #4ade80;
      --error:    #f87171;
      --radius:   10px;
      --mono:     'Space Mono', monospace;
      --sans:     'Space Grotesk', sans-serif;
    }

    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: var(--bg);
      color: var(--text);
      font-family: var(--sans);
      min-height: 100vh;
    }

    /* ── Starfield ── */
    #stars { position: fixed; inset: 0; z-index: 0; pointer-events: none; }
    .star {
      position: absolute; background: white; border-radius: 50%; opacity: 0;
      animation: twinkle var(--dur, 4s) var(--delay, 0s) infinite ease-in-out;
    }
    @keyframes twinkle { 0%,100%{opacity:0} 50%{opacity:var(--op,0.6)} }

    /* ── Layout ── */
    .page { position: relative; z-index: 1; max-width: 960px; margin: 0 auto; padding: 0 24px 80px; }

    /* ── Header ── */
    header { text-align: center; padding: 64px 0 48px; }

    .sun-ring {
      display: inline-block; width: 80px; height: 80px; border-radius: 50%;
      background: radial-gradient(circle at 40% 40%, #ffe066, #f5c842 55%, #c49a1a);
      box-shadow: 0 0 0 8px rgba(245,200,66,.15), 0 0 0 20px rgba(245,200,66,.06), 0 0 60px rgba(245,200,66,.3);
      margin: 0 auto 28px;
      animation: pulse-sun 3s ease-in-out infinite;
    }
    @keyframes pulse-sun {
      0%,100% { box-shadow: 0 0 0 8px rgba(245,200,66,.15), 0 0 0 20px rgba(245,200,66,.06), 0 0 60px rgba(245,200,66,.3); }
      50%      { box-shadow: 0 0 0 12px rgba(245,200,66,.2),  0 0 0 28px rgba(245,200,66,.08), 0 0 90px rgba(245,200,66,.4); }
    }

    .wordmark { font-size: 2.8rem; font-weight: 700; letter-spacing: -1px; line-height: 1; }
    .wordmark span { color: var(--gold); }
    .tagline { margin-top: 10px; color: var(--text-dim); font-size: .95rem; letter-spacing: .05em; text-transform: uppercase; }

    /* ── Grid ── */
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 40px; }
    @media (max-width: 640px) { .grid { grid-template-columns: 1fr; } }

    /* ── Card ── */
    .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 24px; }

    .card-title {
      font-size: .7rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase;
      color: var(--muted); margin-bottom: 16px; display: flex; align-items: center; gap: 8px;
    }
    .card-title::before {
      content: ''; display: block; width: 3px; height: 14px;
      background: var(--gold); border-radius: 2px; flex-shrink: 0;
    }

    /* ── Fields ── */
    .field { margin-bottom: 14px; }
    .field label { display: block; font-size: .78rem; font-weight: 500; color: var(--text-dim); margin-bottom: 5px; }
    .field label .unit { font-family: var(--mono); font-size: .7rem; color: var(--muted); margin-left: 4px; }
    .field input, .field select {
      width: 100%; background: var(--surface2); border: 1px solid var(--border); border-radius: 7px;
      color: var(--text); font-family: var(--mono); font-size: .88rem; padding: 9px 12px;
      outline: none; transition: border-color .2s, box-shadow .2s; appearance: none;
    }
    .field input:focus, .field select:focus {
      border-color: var(--gold-dim);
      box-shadow: 0 0 0 3px rgba(245,200,66,.12);
    }
    .field input::placeholder { color: var(--muted); }
    select option { background: var(--surface2); }

    /* ── Toggle ── */
    .toggle-row { display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }
    .toggle-row label { font-size: .78rem; font-weight: 500; color: var(--text-dim); }
    .toggle { position: relative; width: 44px; height: 24px; }
    .toggle input { opacity: 0; width: 0; height: 0; }
    .slider { position: absolute; inset: 0; background: var(--border); border-radius: 24px; cursor: pointer; transition: background .2s; }
    .slider::after { content: ''; position: absolute; left: 3px; top: 3px; width: 18px; height: 18px; background: white; border-radius: 50%; transition: transform .2s; }
    .toggle input:checked + .slider { background: var(--gold); }
    .toggle input:checked + .slider::after { transform: translateX(20px); }

    /* ── Button ── */
    .btn {
      display: block; width: 100%; margin-top: 8px; padding: 14px;
      background: var(--gold); color: #0b0f1a; font-family: var(--sans);
      font-size: 1rem; font-weight: 700; letter-spacing: .03em;
      border: none; border-radius: var(--radius); cursor: pointer;
      transition: background .15s, transform .1s, box-shadow .2s;
    }
    .btn:hover  { background: #ffe066; box-shadow: 0 0 24px rgba(245,200,66,.35); }
    .btn:active { transform: scale(.98); }
    .btn:disabled { background: var(--muted); cursor: not-allowed; box-shadow: none; }

    /* ── Result Panel ── */
    .result-panel {
      grid-column: 1 / -1; background: var(--surface); border: 1px solid var(--border);
      border-radius: var(--radius); padding: 32px; display: none; animation: fadeUp .3s ease;
    }
    .result-panel.visible { display: block; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }

    .result-top { display: flex; align-items: flex-end; gap: 16px; flex-wrap: wrap; }
    .result-label { font-size: .7rem; font-weight: 600; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); margin-bottom: 6px; }
    .result-value { font-family: var(--mono); font-size: 3.6rem; font-weight: 700; color: var(--gold); line-height: 1; }
    .result-unit  { font-size: 1.1rem; color: var(--text-dim); margin-bottom: 6px; }

    /* ── Energy Bar ── */
    .energy-bar-wrap { margin-top: 28px; }
    .bar-meta { display: flex; justify-content: space-between; font-size: .75rem; color: var(--muted); margin-bottom: 7px; }
    .bar-track { height: 10px; background: var(--surface2); border-radius: 10px; overflow: hidden; border: 1px solid var(--border); }
    .bar-fill  { height: 100%; background: linear-gradient(90deg, var(--gold-dim), var(--gold), #ffe066); border-radius: 10px; width: 0; transition: width 1s cubic-bezier(.4,0,.2,1); }

    /* ── Insight Chips ── */
    .insights { display: flex; flex-wrap: wrap; gap: 10px; margin-top: 24px; }
    .chip { background: var(--surface2); border: 1px solid var(--border); border-radius: 6px; padding: 7px 12px; font-size: .78rem; color: var(--text-dim); display: flex; align-items: center; gap: 6px; }
    .chip .dot { width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0; }
    .dot-good { background: var(--success); }
    .dot-warn { background: var(--gold); }
    .dot-bad  { background: var(--error); }

    /* ── Error ── */
    .error-msg {
      grid-column: 1 / -1; background: rgba(248,113,113,.08); border: 1px solid rgba(248,113,113,.3);
      border-radius: var(--radius); padding: 16px 20px; color: var(--error); font-size: .88rem; display: none;
    }
    .error-msg.visible { display: block; }

    /* ── Spinner ── */
    .spinner {
      display: inline-block; width: 16px; height: 16px;
      border: 2px solid rgba(11,15,26,.3); border-top-color: #0b0f1a;
      border-radius: 50%; animation: spin .6s linear infinite;
      vertical-align: middle; margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    /* ── Footer ── */
    footer { text-align: center; padding-top: 48px; color: var(--muted); font-size: .78rem; font-family: var(--mono); }
    footer a { color: var(--sky-dim); text-decoration: none; }
  </style>
</head>
<body>

<div id="stars"></div>

<div class="page">

  <!-- HEADER -->
  <header>
    <div class="sun-ring"></div>
    <div class="wordmark">Solar<span>IQ</span></div>
    <div class="tagline">Random Forest &middot; Bangladesh Solar Farm &middot; Energy Prediction</div>
  </header>

  <!-- FORM GRID -->
  <div class="grid">

    <!-- SOLAR CONDITIONS -->
    <div class="card">
      <div class="card-title">Solar Conditions</div>

      <div class="field">
        <label>GHI &mdash; Global Horizontal Irradiance <span class="unit">W/m&sup2;</span></label>
        <input type="number" id="GHI" placeholder="e.g. 120.5" step="0.1" min="0" value="120.5" />
      </div>

      <div class="field">
        <label>Cloud Coverage <span class="unit">%</span></label>
        <input type="number" id="clouds_all" placeholder="0 to 100" min="0" max="100" value="20" />
      </div>

      <div class="field">
        <label>Sunlight Time <span class="unit">seconds</span></label>
        <input type="number" id="sunlightTime" placeholder="e.g. 36000" min="0" value="36000" />
      </div>

      <div class="toggle-row">
        <label>Sun is currently up</label>
        <label class="toggle">
          <input type="checkbox" id="isSun" checked />
          <span class="slider"></span>
        </label>
      </div>
    </div>

    <!-- ATMOSPHERIC CONDITIONS -->
    <div class="card">
      <div class="card-title">Atmospheric Conditions</div>

      <div class="field">
        <label>Temperature <span class="unit">&deg;C</span></label>
        <input type="number" id="temp" placeholder="e.g. 28.0" step="0.1" value="28.0" />
      </div>

      <div class="field">
        <label>Pressure <span class="unit">hPa</span></label>
        <input type="number" id="pressure" placeholder="e.g. 1012" value="1012" />
      </div>

      <div class="field">
        <label>Humidity <span class="unit">%</span></label>
        <input type="number" id="humidity" placeholder="0 to 100" min="0" max="100" value="65" />
      </div>

      <div class="field">
        <label>Wind Speed <span class="unit">m/s</span></label>
        <input type="number" id="wind_speed" placeholder="e.g. 3.2" step="0.1" min="0" value="3.2" />
      </div>

      <div class="field">
        <label>Rainfall (last 1 hr) <span class="unit">mm</span></label>
        <input type="number" id="rain_1h" placeholder="0.0" step="0.1" min="0" value="0" />
      </div>
    </div>

    <!-- TIME & WEATHER -->
    <div class="card">
      <div class="card-title">Time &amp; Weather</div>

      <div class="field">
        <label>Hour of Day <span class="unit">0 to 23</span></label>
        <input type="number" id="hour" placeholder="e.g. 12" min="0" max="23" value="12" />
      </div>

      <div class="field">
        <label>Month <span class="unit">1 to 12</span></label>
        <input type="number" id="month" placeholder="e.g. 6" min="1" max="12" value="6" />
      </div>
    </div>

    <!-- PREDICT BUTTON -->
    <div class="card" style="display:flex;flex-direction:column;justify-content:center;gap:12px;">
      <div class="card-title">Run Prediction</div>
      <p style="font-size:.83rem;color:var(--text-dim);line-height:1.55;">
        Powered by a Random Forest model trained on 196,776 hourly readings
        from solar farms. Fill in weather conditions and click Predict.
      </p>
      <button class="btn" id="predictBtn" onclick="runPrediction()">
        &#9728; Predict Energy Output
      </button>
    </div>

    <!-- ERROR MESSAGE -->
    <div class="error-msg" id="errorMsg"></div>

    <!-- RESULT PANEL -->
    <div class="result-panel" id="resultPanel">
      <div class="result-top">
        <div>
          <div class="result-label">Predicted Energy Output</div>
          <div class="result-value" id="resultValue">0</div>
        </div>
        <div class="result-unit">Watt-hours (Wh)</div>
      </div>

      <div class="energy-bar-wrap">
        <div class="bar-meta">
          <span>0 Wh</span>
          <span id="barPct">0%</span>
          <span>5020 Wh (max)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" id="barFill"></div>
        </div>
      </div>

      <div class="insights" id="insights"></div>
    </div>

  </div><!-- /grid -->

  <footer>
    Solar<span style="color:var(--gold)">IQ</span> &nbsp;&middot;&nbsp;
    API: <a href="http://localhost:5000/health">localhost:5000</a> &nbsp;&middot;&nbsp;
    Model: Random Forest &nbsp;&middot;&nbsp;
    Dataset: Bangladesh Solar Farm
  </footer>

</div><!-- /page -->

<script>
  // ── Generate starfield background ─────────────────────────────
  const starsEl = document.getElementById('stars');
  for (let i = 0; i < 90; i++) {
    const s = document.createElement('div');
    s.className = 'star';
    const sz = Math.random() * 2 + 1;
    s.style.cssText = `
      width:${sz}px;height:${sz}px;
      left:${Math.random()*100}%;top:${Math.random()*100}%;
      --dur:${3 + Math.random()*5}s;
      --delay:${Math.random()*6}s;
      --op:${0.3 + Math.random()*0.5};
    `;
    starsEl.appendChild(s);
  }

  // ── Read form field value ──────────────────────────────────────
  function getVal(id) {
    const el = document.getElementById(id);
    if (!el) return 0;
    if (el.type === 'checkbox') return el.checked ? 1 : 0;
    const v = parseFloat(el.value);
    return isNaN(v) ? 0 : v;
  }

  // ── Send prediction request ────────────────────────────────────
  async function runPrediction() {
    const btn    = document.getElementById('predictBtn');
    const errEl  = document.getElementById('errorMsg');
    const panel  = document.getElementById('resultPanel');

    // Hide old results
    errEl.classList.remove('visible');
    panel.classList.remove('visible');

    // Show loading state on button
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';

    // Build the payload — EXACTLY matches FEATURE_COLUMNS in app.py
    const payload = {
      GHI:         getVal('GHI'),
      temp:        getVal('temp'),
      pressure:    getVal('pressure'),
      humidity:    getVal('humidity'),
      wind_speed:  getVal('wind_speed'),
      rain_1h:     getVal('rain_1h'),
      clouds_all:  getVal('clouds_all'),
      isSun:       getVal('isSun'),
      sunlightTime:getVal('sunlightTime'),
      hour:        getVal('hour'),
      month:       getVal('month')
    };

    try {
      // Call the Flask /predict endpoint
      const res  = await fetch('/predict', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify(payload)
      });

      const data = await res.json();

      if (!res.ok || data.error) {
        throw new Error(data.error || 'Prediction failed');
      }

      // Show the result
      showResult(data.predicted_energy_Wh, payload);

    } catch (err) {
      errEl.textContent = 'Error: ' + (err.message || 'Could not reach the API. Is app.py running?');
      errEl.classList.add('visible');
    } finally {
      // Restore button
      btn.disabled = false;
      btn.innerHTML = '&#9728; Predict Energy Output';
    }
  }

  // ── Display result + insight chips ────────────────────────────
  function showResult(wh, p) {
    const MAX = 5020;

    // Show the number
    document.getElementById('resultValue').textContent =
      wh.toLocaleString('en-US', { maximumFractionDigits: 1 });

    // Animate the progress bar
    const pct = Math.min(wh / MAX * 100, 100);
    document.getElementById('barPct').textContent = pct.toFixed(1) + '%';
    requestAnimationFrame(() => {
      setTimeout(() => {
        document.getElementById('barFill').style.width = pct + '%';
      }, 50);
    });

    // Build insight chips
    const chips = [];

    // GHI insight
    if (p.GHI > 100)
      chips.push({ cls: 'dot-good', text: 'GHI ' + p.GHI + ' W/m² — strong solar input' });
    else if (p.GHI > 30)
      chips.push({ cls: 'dot-warn', text: 'GHI ' + p.GHI + ' W/m² — moderate irradiance' });
    else
      chips.push({ cls: 'dot-bad',  text: 'GHI ' + p.GHI + ' W/m² — low solar radiation' });

    // Cloud cover insight
    if (p.clouds_all > 70)
      chips.push({ cls: 'dot-bad',  text: p.clouds_all + '% cloud cover — heavy attenuation' });
    else if (p.clouds_all > 30)
      chips.push({ cls: 'dot-warn', text: p.clouds_all + '% cloud cover — partial shading' });
    else
      chips.push({ cls: 'dot-good', text: p.clouds_all + '% cloud cover — clear sky' });

    // Hour insight
    if (p.hour >= 10 && p.hour <= 14)
      chips.push({ cls: 'dot-good', text: 'Hour ' + p.hour + ':00 — peak solar window' });
    else if ((p.hour >= 7 && p.hour < 10) || (p.hour > 14 && p.hour <= 17))
      chips.push({ cls: 'dot-warn', text: 'Hour ' + p.hour + ':00 — off-peak hours' });
    else
      chips.push({ cls: 'dot-bad',  text: 'Hour ' + p.hour + ':00 — nighttime / no generation' });

    // Sun up/down
    if (p.isSun === 1)
      chips.push({ cls: 'dot-good', text: 'Sun is above the horizon' });
    else
      chips.push({ cls: 'dot-bad',  text: 'Sun is below the horizon' });

    // Render chips
    document.getElementById('insights').innerHTML = chips
      .map(c => '<div class="chip"><span class="dot ' + c.cls + '"></span>' + c.text + '</div>')
      .join('');

    // Show result panel and scroll to it
    document.getElementById('resultPanel').classList.add('visible');
    document.getElementById('resultPanel').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
</script>

</body>
</html>"""


# ── Start server ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 52)
    print("  SolarIQ - Solar Energy Prediction")
    print("=" * 52)
    print("  Dashboard : http://localhost:5000/")
    print("  Health    : http://localhost:5000/health")
    print("  Predict   : POST http://localhost:5000/predict")
    print("=" * 52)
    print("  Press Ctrl+C to stop the server")
    print("=" * 52)
    app.run(debug=True, port=5000)