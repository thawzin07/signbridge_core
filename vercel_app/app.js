const CLASSES = ["A","B","C","D","E","F","G","H","I","K","L","O","P","Q","R","U","V","W","X","Y"];
const $ = (id) => document.getElementById(id);
const els = {
  video: $("video"), still: $("stillPreview"), empty: $("emptyPreview"), consent: $("consent"),
  start: $("startButton"), stop: $("stopButton"), input: $("imageInput"), inputStatus: $("inputStatus"),
  crop: $("cropScale"), cropValue: $("cropValue"), cropGuide: $("cropGuide"), previewShade: $("previewShade"),
  liveState: $("liveState"), predictionLetter: $("predictionLetter"), predictionState: $("predictionState"),
  predictionBox: $("predictionBox"), top3: $("top3List"), add: $("addButton"), sequence: $("sequence"),
  serviceDot: $("serviceDot"), serviceStatus: $("serviceStatus"),
};

let stream = null;
let loopRunning = false;
let stillImage = null;
let stableLetter = null;
let votes = [];
let requestBusy = false;
let lastPredictionAt = 0;

function renderClassChips() {
  const target = $("classChips");
  target.replaceChildren(...CLASSES.map((letter) => {
    const chip = document.createElement("span");
    chip.className = "class-chip";
    chip.textContent = letter;
    return chip;
  }));
}

function setState(text, kind = "") {
  els.predictionState.textContent = text;
  els.predictionBox.classList.toggle("has-letter", Boolean(kind === "ready"));
}

function setCurrent(letter, score, state) {
  els.predictionLetter.textContent = letter || "—";
  setState(state, letter ? "ready" : "");
  els.add.disabled = !letter || !els.consent.checked;
  stableLetter = letter || null;
}

function renderTop3(rows = []) {
  if (!rows.length) {
    const empty = document.createElement("div");
    empty.className = "top3-empty";
    empty.textContent = "Waiting for an image";
    els.top3.replaceChildren(empty);
    return;
  }
  els.top3.replaceChildren(...rows.map(({ letter, score }) => {
    const row = document.createElement("div");
    row.className = "score-row";
    const label = document.createElement("span");
    label.className = "score-letter";
    label.textContent = letter;
    const track = document.createElement("span");
    track.className = "score-track";
    const fill = document.createElement("span");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.className = "score-value";
    value.textContent = `${(score * 100).toFixed(1)}%`;
    row.append(label, track, value);
    return row;
  }));
}

function updateConsent() {
  const allowed = els.consent.checked;
  els.start.disabled = !allowed || !navigator.mediaDevices?.getUserMedia;
  $("uploadLabel").classList.toggle("hidden", !allowed);
  if (!allowed) {
    els.add.disabled = true;
    els.inputStatus.textContent = "Agree to image processing to enable camera and photo testing.";
  } else {
    els.inputStatus.textContent = "Images are processed for inference and not stored by this app.";
  }
}

function captureCrop(source) {
  const width = source.videoWidth || source.naturalWidth;
  const height = source.videoHeight || source.naturalHeight;
  if (!width || !height) throw new Error("The image is not ready yet.");
  const side = Math.min(width, height) * (Number(els.crop.value) / 100);
  const left = (width - side) / 2;
  const top = (height - side) / 2;
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 320;
  const context = canvas.getContext("2d", { alpha: false });
  context.drawImage(source, left, top, side, side, 0, 0, 320, 320);
  return canvas.toDataURL("image/jpeg", 0.74);
}

function updateCropGuide(source = els.video) {
  const box = $("previewWrap");
  const width = box.clientWidth;
  const height = box.clientHeight;
  const mediaWidth = source.videoWidth || source.naturalWidth || width;
  const mediaHeight = source.videoHeight || source.naturalHeight || height;
  const containedWidth = Math.min(width, height * mediaWidth / mediaHeight);
  const containedHeight = Math.min(height, width * mediaHeight / mediaWidth);
  const size = Math.min(containedWidth, containedHeight) * (Number(els.crop.value) / 100);
  els.cropGuide.style.width = `${(size / width) * 100}%`;
}

async function predictSource(source, isLive = false) {
  if (!els.consent.checked) throw new Error("Please agree to image processing before sending an image.");
  const image = captureCrop(source);
  const response = await fetch("/api/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-SignBridge-Consent": "1" },
    cache: "no-store",
    body: JSON.stringify({ image, cropScale: 1, centerX: 0.5, centerY: 0.5 }),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.error || `Model service returned ${response.status}.`);
  renderTop3(data.top3 || []);
  const now = performance.now();
  if (isLive) {
    votes.push(data.ok ? data.letter : null);
    votes = votes.slice(-5);
    const counts = new Map();
    votes.filter(Boolean).forEach((letter) => counts.set(letter, (counts.get(letter) || 0) + 1));
    const winner = [...counts.entries()].sort((a,b) => b[1] - a[1])[0];
    const consensus = winner && winner[1] >= 4 ? winner[0] : null;
    setCurrent(consensus, consensus ? data.score : null, consensus ? `Stable across recent frames: ${consensus}` : "Hold one handshape steady for a stable result");
  } else {
    votes = [];
    setCurrent(data.letter, data.score, data.state || "Prediction available.");
  }
  if (data.ok && data.letter && !stableLetter && isLive) {
    setState("Hold one handshape steady for a stable result");
  }
  lastPredictionAt = now;
}

async function startCamera() {
  if (!els.consent.checked) return;
  if (!navigator.mediaDevices?.getUserMedia) {
    els.inputStatus.textContent = "Camera access needs a secure HTTPS page and browser permission. Choose a photo instead.";
    return;
  }
  try {
    stopCamera();
    stillImage = null;
    els.still.hidden = true;
    els.video.hidden = false;
    els.empty.hidden = true;
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 24, max: 30 } },
    });
    els.video.srcObject = stream;
    await els.video.play();
    updateCropGuide(els.video);
    els.start.disabled = true;
    els.stop.disabled = false;
    els.liveState.innerHTML = "<span></span> Live preview";
    els.liveState.classList.add("active");
    els.cropGuide.hidden = false;
    loopRunning = true;
    votes = [];
    setCurrent(null, null, "Waiting for a stable handshape");
    runInferenceLoop();
  } catch (error) {
    els.inputStatus.textContent = error.name === "NotAllowedError" ? "Camera permission was denied. You can choose a photo instead." : `Could not start camera: ${error.message}`;
    stopCamera();
  }
}

function stopCamera() {
  loopRunning = false;
  if (stream) stream.getTracks().forEach((track) => track.stop());
  stream = null;
  els.video.srcObject = null;
  els.stop.disabled = true;
  els.start.disabled = !els.consent.checked || !navigator.mediaDevices?.getUserMedia;
  els.liveState.innerHTML = "<span></span> Camera off";
  els.liveState.classList.remove("active");
  els.cropGuide.hidden = true;
  if (!stillImage) els.empty.hidden = false;
}

async function runInferenceLoop() {
  while (loopRunning) {
    const began = performance.now();
    if (!requestBusy && els.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
      requestBusy = true;
      try {
        await predictSource(els.video, true);
        els.inputStatus.textContent = `Live inference · ${((performance.now() - began) / 1000).toFixed(2)} s round trip. Preview runs locally.`;
      } catch (error) {
        els.inputStatus.textContent = error.message;
        setCurrent(null, null, "Inference unavailable");
      } finally {
        requestBusy = false;
      }
    }
    await new Promise((resolve) => setTimeout(resolve, Math.max(120, 420 - (performance.now() - began))));
  }
}

async function checkService() {
  try {
    const response = await fetch("/api/predict", { cache: "no-store" });
    const data = await response.json();
    if (!response.ok || !data.ok) throw new Error("Model service unavailable");
    els.serviceDot.classList.add("ready");
    els.serviceStatus.textContent = "Model service ready · 20 classes";
  } catch {
    els.serviceDot.classList.add("error");
    els.serviceStatus.textContent = "Model service is starting or unavailable";
  }
}

els.consent.addEventListener("change", updateConsent);
els.start.addEventListener("click", startCamera);
els.stop.addEventListener("click", stopCamera);
els.crop.addEventListener("input", () => {
  els.cropValue.textContent = `${els.crop.value}%`;
  updateCropGuide(stillImage || els.video);
});
els.add.addEventListener("click", () => {
  if (!stableLetter || !els.consent.checked) return;
  els.sequence.value += stableLetter;
  els.sequence.dispatchEvent(new Event("input", { bubbles: true }));
});
$("clearSequence").addEventListener("click", () => { els.sequence.value = ""; setCurrent(null, null, "Sequence cleared"); });
els.input.addEventListener("change", async () => {
  const file = els.input.files?.[0];
  if (!file) return;
  if (!els.consent.checked) { els.inputStatus.textContent = "Agree to image processing before testing this photo."; return; }
  if (!file.type.startsWith("image/")) { els.inputStatus.textContent = "Choose an image file."; return; }
  if (file.size > 12 * 1024 * 1024) { els.inputStatus.textContent = "Choose a photo smaller than 12 MB."; return; }
  stopCamera();
  try {
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = async () => {
      stillImage = image;
      els.video.hidden = true;
      els.still.src = url;
      els.still.hidden = false;
      els.empty.hidden = true;
      els.cropGuide.hidden = false;
      updateCropGuide(image);
      setCurrent(null, null, "Analyzing photo…");
      els.inputStatus.textContent = "Sending this photo to the model service for one prediction…";
      try { await predictSource(image, false); els.inputStatus.textContent = "Photo processed for inference; the app does not store it."; }
      catch (error) { els.inputStatus.textContent = error.message; setCurrent(null, null, "Inference unavailable"); }
      URL.revokeObjectURL(url);
    };
    image.onerror = () => { URL.revokeObjectURL(url); els.inputStatus.textContent = "This image could not be opened."; };
    image.src = url;
  } catch (error) { els.inputStatus.textContent = error.message; }
});

renderClassChips();
updateConsent();
checkService();
