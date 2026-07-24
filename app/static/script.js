const form = document.getElementById("predict-form");
const resultBox = document.getElementById("result");
const errorBox = document.getElementById("error");
const submitButton = form.querySelector("button");

form.addEventListener("submit", async function (event) {
  // Stops the browser's default behavior, which would reload the page.
  // Without this, the form submission would refresh the page instead
  // of letting our JavaScript handle it.
  event.preventDefault();

  resultBox.classList.add("hidden");
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

  } catch (err) {
    errorBox.textContent = `Error: ${err.message}`;
    errorBox.classList.remove("hidden");

  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "Predict Math Score";
  }
});