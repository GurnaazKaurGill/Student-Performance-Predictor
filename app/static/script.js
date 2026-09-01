/* =========================================================
   Navigation
========================================================= */
const navItems = document.querySelectorAll(".nav-item");
const views = document.querySelectorAll(".view");
const loadedViews = new Set();

function showView(name) {
  navItems.forEach((btn) => btn.classList.toggle("active", btn.dataset.view === name));
  views.forEach((view) => view.classList.toggle("active", view.id === `view-${name}`));

  // Each data-backed view fetches on first visit only, not on every
  // click, so switching tabs afterward is instant.
  if (!loadedViews.has(name)) {
    loadedViews.add(name);
    if (name === "model") loadModelPerformance();
    if (name === "fairness") loadFairnessAudit();
    if (name === "dataset") loadDatasetOverview();
  }
}

navItems.forEach((btn) => {
  btn.addEventListener("click", () => showView(btn.dataset.view));
});

/* =========================================================
   Predict form
========================================================= */
const form = document.getElementById("predict-form");
const submitButton = form.querySelector("button[type='submit']");

const resultEmpty = document.getElementById("result-empty");
const resultBox = document.getElementById("result");
const scoreNumber = document.getElementById("score-number");

const explanationBox = document.getElementById("explanation");
const baseValueSpan = document.getElementById("base-value");
const barsContainer = document.getElementById("bars");

const errorBox = document.getElementById("error");

function renderBars(container, contributions) {
  container.innerHTML = "";
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 0.01);

  contributions.forEach((c, index) => {
    const isPositive = c.contribution >= 0;
    const widthPercent = (Math.abs(c.contribution) / maxAbs) * 50;
    const sign = isPositive ? "+" : "";

    const row = document.createElement("div");
    row.className = "bar-row";
    row.style.animationDelay = `${index * 55}ms`;

    row.innerHTML = `
      <div class="bar-label-row">
        <span class="bar-label">${c.feature}</span>
        <span class="bar-value ${isPositive ? "positive" : "negative"}">${sign}${c.contribution.toFixed(2)}</span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isPositive ? "positive" : "negative"}"></div>
      </div>
    `;

    container.appendChild(row);
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${widthPercent}%`;
    });
  });
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.classList.toggle("loading", isLoading);
  submitButton.querySelector(".btn-label").textContent = isLoading
    ? "Predicting…"
    : "Predict math score";
}

form.addEventListener("submit", async function (event) {
  event.preventDefault();

  resultEmpty.classList.add("hidden");
  resultBox.classList.add("hidden");
  explanationBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  setLoading(true);

  const payload = {
    gender: document.getElementById("gender").value,
    "race/ethnicity": document.getElementById("race").value,
    parental_level_of_education: document.getElementById("education").value,
    lunch: document.getElementById("lunch").value,
    test_preparation_course: document.getElementById("prep").value,
  };

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail ? JSON.stringify(data.detail) : "Prediction failed");
    }

    scoreNumber.textContent = data.predicted_math_score;
    resultBox.classList.remove("hidden");

    baseValueSpan.textContent = data.base_value;
    renderBars(barsContainer, data.contributions);
    explanationBox.classList.remove("hidden");

  } catch (err) {
    errorBox.textContent = `Something went wrong: ${err.message}`;
    errorBox.classList.remove("hidden");

  } finally {
    setLoading(false);
  }
});

/* ---------------------------------------------------------
   Presets
   Gender and race/ethnicity are held constant across both
   presets, on purpose: the point is to isolate the effect of
   preparation and support factors, not to imply anything about
   demographic traits themselves.
--------------------------------------------------------- */
const PRESETS = {
  high: {
    gender: "female",
    race: "group C",
    education: "master's degree",
    lunch: "standard",
    prep: "completed",
  },
  low: {
    gender: "female",
    race: "group C",
    education: "some high school",
    lunch: "free/reduced",
    prep: "none",
  },
};

document.querySelectorAll(".preset-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const preset = PRESETS[btn.dataset.preset];
    document.getElementById("gender").value = preset.gender;
    document.getElementById("race").value = preset.race;
    document.getElementById("education").value = preset.education;
    document.getElementById("lunch").value = preset.lunch;
    document.getElementById("prep").value = preset.prep;
    form.requestSubmit();
  });
});

/* =========================================================
   Model Performance view
========================================================= */
async function loadModelPerformance() {
  const container = document.getElementById("model-content");

  try {
    const response = await fetch("/model-info");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Failed to load model metrics");

    const before = data.before_leakage.linear_regression.rmse;
    const after = data.after_leakage_fix["Linear Regression"].rmse;

    const models = { ...data.after_leakage_fix, "Random Forest (Tuned)": data.tuned_random_forest };
    const maxRmse = Math.max(...Object.values(models).map((m) => m.rmse));

    const compareRows = Object.entries(models).map(([name, m]) => {
      const isFinal = name.toLowerCase() === data.final_model.toLowerCase();
      const widthPercent = (m.rmse / maxRmse) * 100;
      return `
        <div class="compare-row">
          <div class="compare-label-row">
            <span class="compare-label ${isFinal ? "is-final" : ""}">${name}</span>
            <span class="compare-value">RMSE ${m.rmse.toFixed(2)} · MAE ${m.mae.toFixed(2)}</span>
          </div>
          <div class="compare-track">
            <div class="compare-fill ${isFinal ? "is-final" : ""}" data-width="${widthPercent}"></div>
          </div>
        </div>
      `;
    }).join("");

    container.innerHTML = `
      <div class="callout">
        <div class="callout-stat">
          <div class="callout-number">${before}</div>
          <div class="callout-label">Before fix (RMSE)</div>
        </div>
        <span class="callout-arrow">→</span>
        <div class="callout-stat">
          <div class="callout-number">${after}</div>
          <div class="callout-label">After fix (RMSE)</div>
        </div>
        <p class="callout-text">
          An earlier version used the student's reading and writing scores as
          features. Since those come from the same exam sitting as the
          target, this leaked the answer rather than predicting it. Removing
          them raised RMSE from ${before} to ${after} — worse-looking, and
          genuinely honest.
        </p>
      </div>

      <div class="panel">
        <h2>Model comparison</h2>
        <p class="panel-note">Evaluated on the same held-out test set (${data.test_set_size} students), after the leakage fix.</p>
        ${compareRows}
      </div>
    `;

    container.querySelectorAll(".compare-fill").forEach((el) => {
      requestAnimationFrame(() => { el.style.width = `${el.dataset.width}%`; });
    });

  } catch (err) {
    container.innerHTML = `<div class="error">${err.message}</div>`;
  }
}

/* =========================================================
   Fairness Audit view
========================================================= */
async function loadFairnessAudit() {
  const container = document.getElementById("fairness-content");

  try {
    const response = await fetch("/fairness-audit");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Failed to load fairness audit");

    const groups = {};
    data.rows.forEach((row) => {
      if (!groups[row.group]) groups[row.group] = [];
      groups[row.group].push(row);
    });

    const groupsHtml = Object.entries(groups).map(([groupName, rows]) => {
      const maxAbs = Math.max(...rows.map((r) => Math.abs(r.avg_shap_contribution)), 0.01);
      const barsHtml = rows.map((r) => {
        const isPositive = r.avg_shap_contribution >= 0;
        const widthPercent = (Math.abs(r.avg_shap_contribution) / maxAbs) * 50;
        const sign = isPositive ? "+" : "";
        return `
          <div class="bar-row">
            <div class="bar-label-row">
              <span class="bar-label">${r.category} <span style="color:var(--ink-45); font-weight:400;">(n=${r.count}, RMSE ${r.rmse.toFixed(1)})</span></span>
              <span class="bar-value ${isPositive ? "positive" : "negative"}">${sign}${r.avg_shap_contribution.toFixed(2)}</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill ${isPositive ? "positive" : "negative"}" data-width="${widthPercent}"></div>
            </div>
          </div>
        `;
      }).join("");

      return `
        <div class="panel fairness-group">
          <h2>${groupName}</h2>
          ${barsHtml}
        </div>
      `;
    }).join("");

    container.innerHTML = groupsHtml;

    container.querySelectorAll(".bar-fill").forEach((el, i) => {
      setTimeout(() => { el.style.width = `${el.dataset.width}%`; }, i * 25);
    });

  } catch (err) {
    container.innerHTML = `<div class="error">${err.message}</div>`;
  }
}

/* =========================================================
   Dataset Overview view
========================================================= */
async function loadDatasetOverview() {
  const container = document.getElementById("dataset-content");

  try {
    const response = await fetch("/dataset-info");
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Failed to load dataset info");

    function distRows(counts) {
      const max = Math.max(...Object.values(counts));
      return Object.entries(counts).map(([name, count]) => {
        const widthPercent = (count / max) * 100;
        return `
          <div class="dist-row">
            <span class="dist-name">${name}</span>
            <div class="dist-track"><div class="dist-fill" data-width="${widthPercent}"></div></div>
            <span class="dist-count">${count}</span>
          </div>
        `;
      }).join("");
    }

    container.innerHTML = `
      <div class="stat-grid">
        <div class="stat-card"><div class="stat-value">${data.total_students}</div><div class="stat-label">Total students</div></div>
        <div class="stat-card"><div class="stat-value">${data.average_math_score}</div><div class="stat-label">Avg. math score</div></div>
        <div class="stat-card"><div class="stat-value">${data.average_reading_score}</div><div class="stat-label">Avg. reading score</div></div>
        <div class="stat-card"><div class="stat-value">${data.average_writing_score}</div><div class="stat-label">Avg. writing score</div></div>
      </div>

      <div class="panel">
        <h2>Gender</h2>
        ${distRows(data.gender)}
      </div>
      <div class="panel">
        <h2>Race / ethnicity</h2>
        ${distRows(data.race_ethnicity)}
      </div>
      <div class="panel">
        <h2>Parental level of education</h2>
        ${distRows(data.parental_level_of_education)}
      </div>
      <div class="panel">
        <h2>Lunch type</h2>
        ${distRows(data.lunch)}
      </div>
      <div class="panel">
        <h2>Test preparation course</h2>
        ${distRows(data.test_preparation_course)}
      </div>
    `;

    container.querySelectorAll(".dist-fill").forEach((el, i) => {
      setTimeout(() => { el.style.width = `${el.dataset.width}%`; }, i * 20);
    });

  } catch (err) {
    container.innerHTML = `<div class="error">${err.message}</div>`;
  }
}