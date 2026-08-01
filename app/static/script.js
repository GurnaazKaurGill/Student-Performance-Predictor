const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const explanationBox = document.getElementById("explanation");
const baseValueSpan = document.getElementById("base-value");
const barsContainer = document.getElementById("bars");
const errorBox = document.getElementById("error");
const submitButton = form.querySelector("button");

function renderBars(contributions) {
  barsContainer.innerHTML = "";

  // Scale every bar relative to the single largest contribution in
  // this result, so the strongest factor visually fills the row and
  // everything else is sized proportionally against it.
  const maxAbs = Math.max(...contributions.map((c) => Math.abs(c.contribution)), 0.01);

  contributions.forEach((c) => {
    const isPositive = c.contribution >= 0;
    const widthPercent = (Math.abs(c.contribution) / maxAbs) * 50; // 50% = full half-track

    const row = document.createElement("div");
    row.className = "bar-row";

    row.innerHTML = `
      <div class="bar-label-row">
        <span class="bar-label">${c.feature}</span>
        <span class="bar-value ${isPositive ? "positive" : "negative"}">
          ${isPositive ? "+" : ""}${c.contribution.toFixed(2)}
        </span>
      </div>
      <div class="bar-track">
        <div class="bar-fill ${isPositive ? "positive" : "negative"}" style="width: 0%"></div>
      </div>
    `;

    barsContainer.appendChild(row);

    // Set width after insertion so the CSS transition actually animates
    // from 0 to its target width, instead of appearing instantly.
    requestAnimationFrame(() => {
      row.querySelector(".bar-fill").style.width = `${widthPercent}%`;
    });
  });
}

form.addEventListener("submit", async function (event) {
  // Stops the browser's default behavior, which would reload the page.
  // Without this, the form submission would refresh the page instead
  // of letting our JavaScript handle it.
  event.preventDefault();

  resultBox.classList.add("hidden");
  explanationBox.classList.add("hidden");
  errorBox.classList.add("hidden");
  submitButton.disabled = true;
  submitButton.textContent = "Predicting...";

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

    resultBox.textContent = `Predicted math score: ${data.predicted_math_score}`;
    resultBox.classList.remove("hidden");

    baseValueSpan.textContent = data.base_value;
    renderBars(data.contributions);
    explanationBox.classList.remove("hidden");

  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove("hidden");

  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict Math Score";
  }
});