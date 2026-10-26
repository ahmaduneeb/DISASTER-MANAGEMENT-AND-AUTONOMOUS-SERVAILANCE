/* ==========================================================================
   AEGIS dashboard controller
   ========================================================================== */
(() => {
  "use strict";

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const DEFAULTS = window.APP_DEFAULTS || {};
  const PRESETS = window.WEIGHT_PRESETS || {};

  const form = $("#runForm");
  const runBtn = $("#runBtn");
  const previewBtn = $("#previewBtn");
  const formError = $("#formError");
  const engineStatus = $("#engineStatus");
  const engineStatusText = $("#engineStatusText");

  const runPanel = $("#runPanel");
  const results = $("#results");
  const hero = $("#hero");
  const errorPanel = $("#errorPanel");
  const errorText = $("#errorText");

  const progressBar = $("#progressBar");
  const progressPct = $("#progressPct");
  const genLabel = $("#genLabel");
  const runClock = $("#runClock");
  const runPhase = $("#runPhase");
  const runTitle = $("#runTitle");
  const runHint = $("#runHint");

  const liveBest = $("#liveBest");
  const liveCoverage = $("#liveCoverage");
  const liveEnergy = $("#liveEnergy");
  const liveRisk = $("#liveRisk");
  const chart = $("#liveChart");

  const previewImg = $("#previewImg");
  const previewLoading = $("#previewLoading");
  const previewMeta = $("#previewMeta");

  const weightInputs = $$('input[name^="w_"]', form);
  const presetButtons = $$(".preset", $("#presetGrid"));
  const mutationInput = $("#mutationRate");
  const mutationOut = $("#mutationOut");

  let activeJob = null;
  let pollTimer = null;
  let clockTimer = null;
  let startedAt = 0;
  let lastHistory = [];
  let pollFailures = 0;

  /* ------------------------------------------------------------- utilities */
  const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));
  const pct = (v, digits = 1) => `${(Number(v) * 100).toFixed(digits)}%`;
  const num = (v, digits = 3) => Number(v).toFixed(digits);
  const formatTime = (seconds) => {
    const s = Math.max(0, Math.floor(seconds));
    return `${String(Math.floor(s / 60)).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  };

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function setStatus(kind, text) {
    engineStatus.classList.remove("is-running", "is-error", "is-complete");
    if (kind) engineStatus.classList.add(`is-${kind}`);
    engineStatusText.textContent = text;
  }

  function showFormError(message) {
    formError.textContent = message;
    formError.hidden = false;
  }

  function clearFormError() {
    formError.hidden = true;
    formError.textContent = "";
  }

  function collectPayload() {
    const data = new FormData(form);
    const payload = {};
    for (const [key, value] of data.entries()) payload[key] = value;
    return payload;
  }

  /* ------------------------------------------------------------- weights */
  function updateWeightOutputs() {
    weightInputs.forEach((input) => {
      const output = input.closest(".weight-row")?.querySelector("output");
      if (output) output.textContent = Number(input.value).toFixed(2);
    });
    const active = presetButtons.find((button) => {
      const preset = PRESETS[button.dataset.preset];
      if (!preset) return false;
      return Object.entries(preset).every(
        ([key, value]) => Math.abs(Number(form.elements[`w_${key}`].value) - value) < 0.015
      );
    });
    presetButtons.forEach((button) => button.classList.toggle("active", button === active));
  }

  weightInputs.forEach((input) => input.addEventListener("input", updateWeightOutputs));

  presetButtons.forEach((button) => {
    button.addEventListener("click", () => {
      const preset = PRESETS[button.dataset.preset];
      if (!preset) return;
      Object.entries(preset).forEach(([key, value]) => {
        const input = form.elements[`w_${key}`];
        if (input) input.value = value;
      });
      updateWeightOutputs();
    });
  });

  mutationInput.addEventListener("input", () => {
    mutationOut.textContent = Number(mutationInput.value).toFixed(2);
  });

  /* ------------------------------------------------------------- preview */
  async function refreshPreview() {
    previewLoading.hidden = false;
    previewLoading.textContent = "Generating map…";
    previewImg.classList.remove("loaded");
    try {
      const response = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collectPayload()),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not render the mission map.");
      await new Promise((resolve, reject) => {
        previewImg.onload = resolve;
        previewImg.onerror = reject;
        previewImg.src = `${data.url}?t=${Date.now()}`;
      });
      previewImg.classList.add("loaded");
      previewLoading.hidden = true;
      previewMeta.textContent = `seed ${Number(form.elements.seed.value) || 0}`;
    } catch (error) {
      previewImg.classList.remove("loaded");
      previewLoading.hidden = false;
      previewLoading.textContent = "Preview unavailable";
      previewMeta.textContent = "—";
    }
  }

  const schedulePreview = debounce(refreshPreview, 380);
  ["grid_size", "seed", "n_drones", "n_patrol"].forEach((name) => {
    const input = form.elements[name];
    if (input) {
      input.addEventListener("change", schedulePreview);
      input.addEventListener("input", schedulePreview);
    }
  });
  previewBtn.addEventListener("click", refreshPreview);

  /* -------------------------------------------------------------- chart */
  function drawChart(history) {
    if (!chart) return;
    const rect = chart.getBoundingClientRect();
    if (rect.width < 30) return;
    const dpr = window.devicePixelRatio || 1;
    chart.width = Math.round(rect.width * dpr);
    chart.height = Math.round(rect.height * dpr);
    const ctx = chart.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width;
    const h = rect.height;
    const pad = { left: 42, right: 10, top: 12, bottom: 20 };
    ctx.clearRect(0, 0, w, h);

    const values = history.flatMap((p) => [p.best, p.average].filter((v) => Number.isFinite(v)));
    if (history.length < 2 || values.length < 2) {
      ctx.fillStyle = "rgba(142,164,194,.55)";
      ctx.font = "10px ui-monospace, monospace";
      ctx.textAlign = "center";
      ctx.fillText("Waiting for generations…", w / 2, h / 2);
      return;
    }

    let minY = Math.min(...values);
    let maxY = Math.max(...values);
    if (maxY - minY < 1e-6) { maxY += 0.1; minY -= 0.1; }
    const padY = (maxY - minY) * 0.12;
    minY -= padY;
    maxY += padY;

    const minGen = history[0].generation;
    const maxGen = Math.max(history[history.length - 1].generation, minGen + 1);
    const x = (g) => pad.left + ((g - minGen) / (maxGen - minGen)) * (w - pad.left - pad.right);
    const y = (v) => pad.top + (1 - (v - minY) / (maxY - minY)) * (h - pad.top - pad.bottom);

    ctx.strokeStyle = "rgba(89,124,168,.16)";
    ctx.lineWidth = 1;
    ctx.font = "9px ui-monospace, monospace";
    ctx.fillStyle = "rgba(142,164,194,.55)";
    ctx.textAlign = "right";
    for (let i = 0; i <= 4; i += 1) {
      const value = minY + ((maxY - minY) * i) / 4;
      const yy = y(value);
      ctx.beginPath();
      ctx.moveTo(pad.left, yy);
      ctx.lineTo(w - pad.right, yy);
      ctx.stroke();
      ctx.fillText(value.toFixed(2), pad.left - 6, yy + 3);
    }

    const drawLine = (key, color, width, fill) => {
      ctx.beginPath();
      history.forEach((point, i) => {
        const xx = x(point.generation);
        const yy = y(point[key]);
        if (i === 0) ctx.moveTo(xx, yy);
        else ctx.lineTo(xx, yy);
      });
      if (fill) {
        ctx.lineTo(x(history[history.length - 1].generation), h - pad.bottom);
        ctx.lineTo(x(history[0].generation), h - pad.bottom);
        ctx.closePath();
        const gradient = ctx.createLinearGradient(0, pad.top, 0, h - pad.bottom);
        gradient.addColorStop(0, "rgba(38,217,255,.18)");
        gradient.addColorStop(1, "rgba(38,217,255,0)");
        ctx.fillStyle = gradient;
        ctx.fill();
        ctx.beginPath();
        history.forEach((point, i) => {
          const xx = x(point.generation);
          const yy = y(point[key]);
          if (i === 0) ctx.moveTo(xx, yy);
          else ctx.lineTo(xx, yy);
        });
      }
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.stroke();
    };

    drawLine("average", "#ffb454", 1.4, false);
    drawLine("best", "#26d9ff", 2.0, true);

    const last = history[history.length - 1];
    ctx.beginPath();
    ctx.arc(x(last.generation), y(last.best), 3, 0, Math.PI * 2);
    ctx.fillStyle = "#b9f6ff";
    ctx.shadowColor = "#26d9ff";
    ctx.shadowBlur = 9;
    ctx.fill();
    ctx.shadowBlur = 0;
  }

  if ("ResizeObserver" in window) {
    new ResizeObserver(() => drawChart(lastHistory)).observe(chart);
  }

  /* --------------------------------------------------------------- poll */
  async function pollJob(jobId) {
    if (!activeJob || activeJob !== jobId) return;
    try {
      const response = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!response.ok) throw new Error("Job status unavailable.");
      const job = await response.json();
      pollFailures = 0;
      updateProgress(job);
      if (job.status === "complete") {
        finishRun(job);
        return;
      }
      if (job.status === "error") {
        failRun(job.error || "The optimiser stopped unexpectedly.");
        return;
      }
    } catch (error) {
      pollFailures += 1;
      if (pollFailures >= 4) {
        failRun("Lost contact with the simulation process. Check the terminal for details.");
        return;
      }
    }
    pollTimer = setTimeout(() => pollJob(jobId), 700);
  }

  function updateProgress(job) {
    const progress = clamp(Number(job.progress || 0), 0, 1);
    progressBar.style.width = `${(progress * 100).toFixed(1)}%`;
    progressPct.textContent = `${Math.round(progress * 100)}%`;
    const generations = job.generations || 0;
    genLabel.textContent = `Generation ${job.generation || 0} / ${generations || "—"}`;
    if (job.best !== null && job.best !== undefined) liveBest.textContent = num(job.best, 4);
    if (job.coverage !== null && job.coverage !== undefined) liveCoverage.textContent = pct(job.coverage);
    if (job.energy_frac !== null && job.energy_frac !== undefined) liveEnergy.textContent = pct(job.energy_frac, 0);
    if (job.risk !== null && job.risk !== undefined) liveRisk.textContent = num(job.risk, 3);
    if (job.status === "queued") {
      runPhase.textContent = "Queued";
      runTitle.textContent = "Preparing the population";
      runHint.textContent = "Waiting for a free worker to start the genetic search.";
    } else {
      runPhase.textContent = "Optimising";
      runTitle.textContent = "Evolving fleet strategy";
      runHint.textContent = job.generation
        ? `Evaluating generation ${job.generation}. The best strategy is retained while mutation explores alternatives.`
        : "The engine is evaluating candidate strategies. Figures are rendered when the run completes.";
    }
    lastHistory = Array.isArray(job.history) ? job.history : [];
    drawChart(lastHistory);
  }

  function startClock() {
    startedAt = Date.now();
    runClock.textContent = "00:00";
    clearInterval(clockTimer);
    clockTimer = setInterval(() => {
      runClock.textContent = formatTime((Date.now() - startedAt) / 1000);
    }, 500);
  }

  function stopClock(seconds) {
    clearInterval(clockTimer);
    clockTimer = null;
    if (Number.isFinite(Number(seconds))) runClock.textContent = formatTime(seconds);
  }

  /* --------------------------------------------------------------- run */
  async function startRun() {
    clearFormError();
    errorPanel.hidden = true;
    if (!form.reportValidity()) return;

    const payload = collectPayload();
    runBtn.disabled = true;
    setStatus("running", "Optimising");
    hero.hidden = true;
    results.hidden = true;
    runPanel.hidden = false;
    lastHistory = [];
    liveBest.textContent = "—";
    liveCoverage.textContent = "—";
    liveEnergy.textContent = "—";
    liveRisk.textContent = "—";
    progressBar.style.width = "0%";
    progressPct.textContent = "0%";
    genLabel.textContent = "Preparing population…";
    runPhase.textContent = "Initialising";
    runTitle.textContent = "Seeding the population";
    runHint.textContent = "Building the first generation and evaluating candidate strategies.";
    startClock();
    requestAnimationFrame(() => drawChart([]));

    try {
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Could not start the simulation.");
      activeJob = data.job_id;
      pollJob(activeJob);
    } catch (error) {
      failRun(error.message);
    }
  }

  function failRun(message) {
    stopClock();
    activeJob = null;
    runPanel.hidden = true;
    runBtn.disabled = false;
    hero.hidden = false;
    results.hidden = true;
    errorPanel.hidden = false;
    errorText.textContent = message;
    setStatus("error", "Run failed");
    errorPanel.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    startRun();
  });

  $("#retryBtn").addEventListener("click", () => {
    errorPanel.hidden = true;
    setStatus("", "Engine ready");
    form.requestSubmit();
  });

  /* ------------------------------------------------------------- results */
  function metricCard(label, value, foot, accent, suffix = "") {
    const card = document.createElement("div");
    card.className = "metric";
    card.style.setProperty("--metric-accent", accent);
    const labelEl = document.createElement("div");
    labelEl.className = "metric-label";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "metric-value";
    valueEl.textContent = value;
    if (suffix) {
      const small = document.createElement("small");
      small.textContent = suffix;
      valueEl.appendChild(small);
    }
    const footEl = document.createElement("div");
    footEl.className = "metric-foot";
    footEl.textContent = foot;
    card.append(labelEl, valueEl, footEl);
    return card;
  }

  function renderMetrics(job) {
    const metrics = job.result.metrics;
    const config = job.result.config;
    const grid = $("#metricsGrid");
    grid.replaceChildren(
      metricCard("Best fitness", num(metrics.fitness, 3), "scalarised objective", "#26d9ff"),
      metricCard("Priority coverage", pct(metrics.coverage), "weighted disaster-zone mass", "#26d9ff"),
      metricCard("Target detection", pct(metrics.detection_rate), `p ≥ 0.5 · D = ${num(metrics.detection, 3)}`, "#3ddc97"),
      metricCard("Energy used", pct(metrics.energy_frac, 0), `${num(metrics.energy, 1)} / ${num(job.result.config.mission.energy_budget, 0)} units`, "#ffb454"),
      metricCard("Flight time", pct(metrics.time_frac, 0), `${num(metrics.time, 1)} mission ticks`, "#a78bfa"),
      metricCard("Hazard exposure", num(metrics.risk, 3), "lower is safer", "#ff6482"),
      metricCard("Active fleet", `${metrics.n_active} / ${config.mission.n_drones}`, "UAVs deployed or in reserve", "#3ddc97"),
      metricCard("Pareto archive", `${job.result.pareto_size}`, "non-dominated alternatives", "#8ea4c2")
    );

    const meta = $("#resultMeta");
    meta.replaceChildren();
    const chips = [
      ["Evaluations", Number(job.result.evaluations).toLocaleString()],
      ["Generations", `${job.result.generations_run} / ${config.ga.generations}`],
      ["Termination", job.result.termination.replaceAll("_", " ")],
      ["Grid", `${config.mission.grid_size} × ${config.mission.grid_size}`],
      ["Seed", String(config.mission.seed)],
      ["Runtime", `${Number(job.result.elapsed).toFixed(1)} s`],
    ];
    chips.forEach(([label, value]) => {
      const chip = document.createElement("span");
      chip.className = "meta-chip";
      const b = document.createElement("b");
      b.textContent = label;
      chip.append(b, document.createTextNode(value));
      meta.appendChild(chip);
    });

    $("#resultSubtitle").textContent =
      `Selected from ${job.result.pareto_size} non-dominated strategies on a ${config.mission.grid_size}×${config.mission.grid_size} grid.`;
  }

  function setImage(id, url) {
    const img = $(id);
    if (!img) return;
    img.src = url;
  }

  function finishRun(job) {
    activeJob = null;
    runBtn.disabled = false;
    stopClock(job.result ? job.result.elapsed : 0);
    progressBar.style.width = "100%";
    progressPct.textContent = "100%";
    genLabel.textContent = `Generation ${job.generation} / ${job.generations}`;
    setStatus("complete", "Run complete");
    runPanel.hidden = true;
    errorPanel.hidden = true;
    hero.hidden = true;

    renderMetrics(job);
    const figs = job.result.figures;
    setImage("#figPaths", figs.paths);
    setImage("#figCoverage", figs.coverage);
    setImage("#figConvergence", figs.convergence);
    setImage("#figEvolution", figs.evolution);
    setImage("#figPareto", figs.pareto);
    setImage("#figMission", figs.mission);

    const downloads = job.result.downloads;
    $("#downloadGenome").href = downloads["best_genome.json"];
    $("#downloadSummary").href = downloads["run_summary.json"];
    $("#dlBest").href = downloads["best_genome.json"];
    $("#dlHistory").href = downloads["history.json"];
    $("#dlPareto").href = downloads["pareto_archive.json"];
    $("#dlConfig").href = `/runs/${job.id}/config.json`;

    // Reset to the trajectory tab after each run.
    $$(".tab", $("#resultTabs")).forEach((tab) => tab.classList.toggle("active", tab.dataset.tab === "paths"));
    $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === "paths"));

    results.hidden = false;
    requestAnimationFrame(() => {
      results.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  }

  $("#resultTabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (!tab) return;
    $$(".tab", $("#resultTabs")).forEach((item) => item.classList.toggle("active", item === tab));
    $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.dataset.panel === tab.dataset.tab));
  });

  /* ---------------------------------------------------------------- init */
  updateWeightOutputs();
  const initialSeed = Number(DEFAULTS.seed || form.elements.seed.value || 0);
  previewMeta.textContent = `seed ${initialSeed}`;
  refreshPreview();
})();
