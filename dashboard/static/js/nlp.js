const sentenceInput = document.getElementById("sentence-input");
const predictBtn = document.getElementById("predict-btn");
const predictMeta = document.getElementById("predict-meta");
const predictError = document.getElementById("predict-error");
const barsEl = document.getElementById("predict-bars");
const pplBtn = document.getElementById("ppl-btn");
const pplResult = document.getElementById("ppl-result");
const pplValue = document.getElementById("ppl-value");
const pplFactor = document.getElementById("ppl-factor");

let activeCorrect = null;

document.querySelectorAll(".example-chip").forEach((chip) => {
  chip.addEventListener("click", () => {
    sentenceInput.value = chip.dataset.masked;
    activeCorrect = chip.dataset.correct;
    predictMeta.innerHTML =
      `<span class="original">Original: ${chip.dataset.original}</span> — ` +
      `<span class="correct-tag">expected: ${chip.dataset.correct}</span>`;
    predictMeta.classList.remove("hidden");
    runPredict();
  });
});

predictBtn.addEventListener("click", () => {
  activeCorrect = null;
  predictMeta.classList.add("hidden");
  runPredict();
});

sentenceInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    activeCorrect = null;
    predictMeta.classList.add("hidden");
    runPredict();
  }
});

async function runPredict() {
  const sentence = sentenceInput.value.trim();
  predictError.classList.add("hidden");
  barsEl.innerHTML = "";

  if (!sentence) return;
  if (!sentence.includes("[MASK]")) {
    predictError.textContent = "Please include [MASK] in your sentence.";
    predictError.classList.remove("hidden");
    return;
  }

  predictBtn.disabled = true;
  predictBtn.textContent = "Predicting…";

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sentence }),
    });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || "Prediction failed");
    }
    const data = await res.json();
    renderBars(data.predictions);
  } catch (e) {
    predictError.textContent = e.message;
    predictError.classList.remove("hidden");
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Predict";
  }
}

function renderBars(predictions) {
  barsEl.innerHTML = "";
  predictions.forEach((p, i) => {
    const isCorrect =
      activeCorrect && (p.piece_clean === activeCorrect || p.piece === activeCorrect);
    const row = document.createElement("div");
    row.className = "bar-row";
    row.innerHTML = `
      <span class="bar-rank">#${i + 1}</span>
      <span class="bar-label ${isCorrect ? "correct" : ""}">${escapeHtml(p.piece)}</span>
      <span class="bar-track"><span class="bar-fill ${isCorrect ? "correct" : ""}" style="width:${p.prob}%"></span></span>
      <span class="bar-pct">${p.prob.toFixed(1)}%</span>
    `;
    barsEl.appendChild(row);
  });
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

pplBtn.addEventListener("click", async () => {
  pplBtn.disabled = true;
  pplBtn.textContent = "Computing… (test set, ~500 sentences)";
  try {
    const res = await fetch("/api/perplexity");
    const data = await res.json();
    pplValue.textContent = data.perplexity;
    pplFactor.textContent = data.improvement_factor + "×";
    pplResult.classList.remove("hidden");
  } finally {
    pplBtn.disabled = false;
    pplBtn.textContent = "Recompute";
  }
});
