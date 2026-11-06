// Crystal Studio — interfaccia della preview live.
// Il server è la fonte di verità: ogni modifica va a /api/set, lo stato torna via
// eventi SSE ("config", "status", "render", "toast"). Il video arriva come MJPEG.
"use strict";

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const clone = (o) => JSON.parse(JSON.stringify(o ?? {}));
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const store = {
  get(k, d) { try { const v = localStorage.getItem("crystal." + k); return v === null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("crystal." + k, JSON.stringify(v)); } catch { /* niente */ } },
};

function getPath(obj, key) {
  let o = obj;
  for (const p of key.split(".")) {
    if (o === null || typeof o !== "object" || !(p in o)) return undefined;
    o = o[p];
  }
  return o;
}
const hasPath = (obj, key) => getPath(obj, key) !== undefined;

const S = {
  schema: [], cfg: null, status: null, statusAt: 0, files: {}, bands: null, bandsKey: null,
  undo: [], redo: [], gesture: null, controls: new Map(), dragging: null,
  pin: null, compare: false, split: 0.5, audioOn: store.get("audio", true), audioKey: null,
  scrub: null, render: null,
};

// ═════════════════════════════════════════════════════════════════════════════
//  Rete
// ═════════════════════════════════════════════════════════════════════════════
async function api(cmd, body = {}, { quiet = false } = {}) {
  const r = await fetch("/api/" + cmd, { method: "POST", body: JSON.stringify(body) });
  let j = {};
  try { j = await r.json(); } catch { /* vuoto */ }
  if (!r.ok || j.error) {
    if (!quiet) toast(j.error || `Errore ${r.status}`, "error");
    throw new Error(j.error || r.status);
  }
  if (j.user) applyConfig({ ...S.cfg, ...j });
  return j;
}

// invio "l'ultimo vince": mentre una richiesta per una chiave è in volo, tiene solo il valore più recente
const inflight = new Map();
function sendValue(key, value) {
  const st = inflight.get(key);
  if (st) { st.next = value; st.hasNext = true; return; }
  const s = { next: null, hasNext: false };
  inflight.set(key, s);
  const go = (v) => api("set", { key, value: v }).catch(() => {}).finally(() => {
    if (s.hasNext) { s.hasNext = false; go(s.next); } else inflight.delete(key);
  });
  go(value);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Annulla / ripeti: un'istantanea delle chiavi del preset prima di ogni gesto
// ═════════════════════════════════════════════════════════════════════════════
function beginEdit() { if (!S.gesture && S.cfg) S.gesture = clone(S.cfg.user); }
function endEdit() {
  if (S.gesture) {
    S.undo.push(S.gesture);
    if (S.undo.length > 200) S.undo.shift();
    S.redo = [];
    S.gesture = null;
    updateUndoButtons();
  }
}
async function undo() {
  if (!S.undo.length) return;
  const prev = S.undo.pop();
  S.redo.push(clone(S.cfg.user));
  await api("user", { user: prev }).catch(() => {});
  updateUndoButtons();
}
async function redo() {
  if (!S.redo.length) return;
  const next = S.redo.pop();
  S.undo.push(clone(S.cfg.user));
  await api("user", { user: next }).catch(() => {});
  updateUndoButtons();
}
function updateUndoButtons() {
  $("#undo").disabled = !S.undo.length;
  $("#redo").disabled = !S.redo.length;
}
function setKey(key, value) { beginEdit(); sendValue(key, value); endEdit(); }

// ═════════════════════════════════════════════════════════════════════════════
//  Controlli dei parametri
// ═════════════════════════════════════════════════════════════════════════════
function decimals(step) {
  if (!step || step >= 1) return 0;
  return Math.min(8, Math.max(0, Math.ceil(-Math.log10(step) - 1e-9)));
}
function fmt(v, p) {
  if (typeof v !== "number" || !isFinite(v)) return String(v);
  if (p.type === "int") return String(Math.round(v));
  if (p.log) {
    if (v === 0) return "0";
    const a = Math.abs(v);
    return a >= 1000 || a < 1e-4 ? v.toExponential(2).replace("e-", "e-").replace(/\.?0+e/, "e") : String(+v.toPrecision(3));
  }
  return String(+v.toFixed(decimals(p.step)));
}
function quantize(v, p) {
  if (p.type === "int") return Math.round(v);
  if (p.log) return +v.toPrecision(3);
  const d = decimals(p.step);
  return +(Math.round(v / p.step) * p.step).toFixed(d);
}

function makeNumber(p, row) {
  const ctl = document.createElement("div");
  ctl.className = "p-ctl";
  const range = document.createElement("input");
  range.type = "range"; range.min = 0; range.max = 1000; range.step = 1;
  const num = document.createElement("input");
  num.type = "number"; num.step = p.log ? "any" : p.step;
  ctl.append(range, num);
  let lo = p.min, hi = p.max; // si allargano se il valore esce dal range
  const toPos = (v) => {
    if (p.log) {
      if (v <= 0) return 0;
      return 1000 * (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo));
    }
    return 1000 * (v - lo) / (hi - lo || 1);
  };
  const fromPos = (x) => p.log ? Math.exp(Math.log(lo) + (x / 1000) * (Math.log(hi) - Math.log(lo))) : lo + (x / 1000) * (hi - lo);
  const paint = () => range.style.setProperty("--p", (range.value / 10) + "%");
  const show = (v) => {
    if (typeof v !== "number") return;
    if (v < lo && !(p.log && v <= 0)) lo = v;
    if (v > hi) hi = v;
    range.value = Math.max(0, Math.min(1000, toPos(v)));
    paint();
    if (document.activeElement !== num) num.value = fmt(v, p);
  };
  range.addEventListener("pointerdown", () => { beginEdit(); S.dragging = p.key; });
  range.addEventListener("input", () => {
    beginEdit();
    const v = quantize(fromPos(+range.value), p);
    num.value = fmt(v, p);
    paint();
    sendValue(p.key, v);
  });
  range.addEventListener("change", () => { S.dragging = null; endEdit(); });
  range.addEventListener("keydown", (e) => e.stopPropagation());
  num.addEventListener("focus", beginEdit);
  num.addEventListener("change", () => {
    const v = parseFloat(num.value.replace(",", "."));
    if (!isFinite(v)) return;
    const q = p.type === "int" ? Math.round(v) : v;
    show(q);
    beginEdit(); sendValue(p.key, q); endEdit();
  });
  num.addEventListener("keydown", (e) => { if (e.key === "Enter") num.blur(); e.stopPropagation(); });
  row.append(ctl);
  return { show };
}

function makeSwitch(onChange) {
  const sw = document.createElement("label");
  sw.className = "switch";
  const inp = document.createElement("input");
  inp.type = "checkbox";
  sw.append(inp, document.createElement("i"));
  inp.addEventListener("change", () => onChange(inp.checked));
  inp.addEventListener("click", (e) => e.stopPropagation());
  sw.addEventListener("click", (e) => e.stopPropagation());
  return { el: sw, inp };
}

const toHex = (c) => "#" + c.map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, "0")).join("");
const fromHex = (h) => [1, 3, 5].map((i) => parseInt(h.slice(i, i + 2), 16));

function makeControl(p, row) {
  if (p.type === "float" || p.type === "int") return makeNumber(p, row);
  const ctl = document.createElement("div");
  ctl.className = "p-ctl";
  row.append(ctl);
  if (p.type === "bool") {
    const sw = makeSwitch((v) => setKey(p.key, v));
    ctl.append(sw.el);
    const lab = document.createElement("span");
    lab.className = "dim small";
    ctl.append(lab);
    return { show: (v) => { sw.inp.checked = !!v; lab.textContent = v ? "sì" : "no"; } };
  }
  if (p.type === "choice") {
    const sel = document.createElement("select");
    for (const c of p.choices) sel.append(new Option(c, c));
    sel.addEventListener("change", () => setKey(p.key, sel.value));
    sel.addEventListener("keydown", (e) => e.stopPropagation());
    ctl.append(sel);
    return { show: (v) => { if (![...sel.options].some((o) => o.value === v)) sel.append(new Option(v, v)); sel.value = v; } };
  }
  if (p.type === "color") {
    const col = document.createElement("input");
    col.type = "color";
    const rgb = document.createElement("span");
    rgb.className = "rgb";
    col.addEventListener("input", () => { beginEdit(); const c = fromHex(col.value); rgb.textContent = c.join(", "); sendValue(p.key, c); });
    col.addEventListener("change", endEdit);
    ctl.append(col, rgb);
    return { show: (v) => { if (Array.isArray(v)) { col.value = toHex(v); rgb.textContent = v.join(", "); } } };
  }
  // file e testo: campo libero con suggerimenti (accetta anche glob e liste separate da virgola)
  const inp = document.createElement("input");
  inp.type = "text";
  inp.spellcheck = false;
  if (p.type === "file") inp.setAttribute("list", "dl-" + p.files);
  const parse = (s) => {
    const parts = s.split(",").map((x) => x.trim()).filter(Boolean);
    return Array.isArray(p.default) || parts.length > 1 ? parts : (parts[0] ?? "");
  };
  inp.addEventListener("change", () => setKey(p.key, parse(inp.value)));
  inp.addEventListener("keydown", (e) => { if (e.key === "Enter") inp.blur(); e.stopPropagation(); });
  ctl.append(inp);
  return { show: (v) => { if (document.activeElement !== inp) inp.value = Array.isArray(v) ? v.join(", ") : (v ?? ""); } };
}

function buildPanel() {
  const root = $("#groups");
  root.innerHTML = "";
  const open = store.get("open", { lenses: true, organic: true, blend: true });
  for (const g of S.schema) {
    const det = document.createElement("details");
    det.className = "grp";
    det.dataset.id = g.id;
    det.open = !!open[g.id];
    det.addEventListener("toggle", () => { const o = store.get("open", {}); o[g.id] = det.open; store.set("open", o); });
    const sum = document.createElement("summary");
    const title = document.createElement("span");
    title.className = "grp-title";
    title.textContent = g.title;
    const count = document.createElement("span");
    count.className = "grp-count";
    count.hidden = true;
    sum.append(title, count);
    const enabledP = g.params.find((p) => p.name === "enabled");
    let enabledSw = null;
    if (enabledP) {
      enabledSw = makeSwitch((v) => setKey(enabledP.key, v));
      enabledSw.el.title = "Attiva / disattiva";
      sum.append(enabledSw.el);
    }
    det.append(sum);
    if (g.help) {
      const h = document.createElement("div");
      h.className = "grp-help";
      h.textContent = g.help;
      det.append(h);
    }
    const params = document.createElement("div");
    params.className = "params";
    det.append(params);
    root.append(det);
    for (const p of g.params) {
      if (p === enabledP) {
        S.controls.set(p.key, { p, row: null, show: (v) => { enabledSw.inp.checked = !!v; det.classList.toggle("off", !v); }, group: det });
        continue;
      }
      const row = document.createElement("div");
      row.className = "param";
      row.dataset.key = p.key;
      const head = document.createElement("div");
      head.className = "p-head";
      const name = document.createElement("span");
      name.className = "p-name";
      name.textContent = p.name;
      name.title = `${p.key}\ndefault: ${JSON.stringify(p.default)}${p.help ? "\n\n" + p.help : ""}\n\ndoppio clic: torna al default`;
      const help = document.createElement("span");
      help.className = "p-help";
      help.textContent = p.help;
      help.title = p.help;
      const reset = document.createElement("button");
      reset.className = "p-reset";
      reset.textContent = "↺";
      reset.title = `Togli dal preset (torna al default: ${JSON.stringify(p.default)})`;
      reset.addEventListener("click", () => resetKey(p.key));
      name.addEventListener("dblclick", () => resetKey(p.key));
      head.append(name, help, reset);
      row.append(head);
      const c = makeControl(p, row);
      params.append(row);
      S.controls.set(p.key, { p, row, show: c.show, group: det });
    }
  }
  applyFilter();
}

function resetKey(key) {
  if (!hasPath(S.cfg.user, key)) return;
  beginEdit();
  api("reset", { key }).catch(() => {}).finally(endEdit);
}

function refreshControls() {
  const c = S.cfg;
  if (!c) return;
  const counts = new Map();
  for (const [key, ctl] of S.controls) {
    const v = getPath(c.effective, key);
    if (S.dragging !== key) ctl.show(v);
    const isSet = hasPath(c.user, key);
    const changed = !same(getPath(c.user, key), getPath(c.saved, key));
    if (ctl.row) {
      ctl.row.classList.toggle("set", isSet);
      ctl.row.classList.toggle("changed", changed);
    }
    const id = ctl.group.dataset.id;
    const k = counts.get(id) || { set: 0, changed: 0 };
    if (isSet) k.set++;
    if (changed) k.changed++;
    counts.set(id, k);
  }
  for (const det of $$("details.grp")) {
    const k = counts.get(det.dataset.id) || { set: 0, changed: 0 };
    const el = $(".grp-count", det);
    el.hidden = !k.set && !k.changed;
    el.textContent = k.changed ? `${k.changed} da salvare` : String(k.set);
    el.classList.toggle("changed", !!k.changed);
    el.title = `${k.set} chiavi scritte nel preset` + (k.changed ? `, ${k.changed} cambiate rispetto al file` : "");
  }
  applyFilter();
}

function applyFilter() {
  const q = $("#search").value.trim().toLowerCase();
  const onlySet = $("#only-set").checked;
  for (const det of $$("details.grp")) {
    let any = false;
    for (const row of $$(".param", det)) {
      const ctl = S.controls.get(row.dataset.key);
      const p = ctl.p;
      const hit = !q || p.key.toLowerCase().includes(q) || (p.help || "").toLowerCase().includes(q)
        || ctl.group.querySelector(".grp-title").textContent.toLowerCase().includes(q);
      const ok = hit && (!onlySet || row.classList.contains("set") || row.classList.contains("changed"));
      row.classList.toggle("hidden", !ok);
      any ||= ok;
    }
    det.classList.toggle("hidden", !any);
    if ((q || onlySet) && any) det.open = true;
  }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Stato dal server
// ═════════════════════════════════════════════════════════════════════════════
function applyConfig(c) {
  const prev = S.cfg;
  S.cfg = c;
  refreshControls();
  // preset
  const sel = $("#preset");
  const names = c.presets || [];
  const want = names.includes(c.preset) ? names : [c.preset, ...names];
  if (!same([...sel.options].map((o) => o.value), want)) {
    sel.innerHTML = "";
    for (const n of want) sel.append(new Option(n, n));
  }
  sel.value = c.preset;
  sel.title = c.path || "";
  $("#dirty").hidden = !c.dirty;
  $("#save").disabled = !c.editable || !c.dirty;
  $("#save").title = c.editable ? `Salva in ${c.path} (⌘S)` : "Questo config non è un preset: usa «Salva come»";
  $("#revert").disabled = !c.dirty;
  document.title = `${c.dirty ? "● " : ""}${c.preset} — Crystal Studio`;
  // seed
  if (document.activeElement !== $("#seed")) $("#seed").value = c.seed ?? "";
  $("#seed-fix").classList.toggle("on", !!c.seed_fixed);
  $("#seed-fix").title = c.seed_fixed ? "Il seed è scritto nel preset (clic: toglilo, torna casuale)"
                                      : "Il seed è casuale: clic per scriverlo nel preset";
  // vista
  $("#view-scale").value = String(c.view?.scale ?? 0.5);
  $("#view-fps").value = c.view?.fps ? String(c.view.fps) : "";
  // errore
  const err = $("#error");
  err.hidden = !c.error;
  err.textContent = c.error ? "⚠ " + c.error + "\n(la preview mostra l'ultimo stato valido)" : "";
  if (c.W && (!prev || prev.W !== c.W || prev.H !== c.H)) layoutStage();
  // audio
  const key = c.audio ? c.audio.join("|") : null;
  if (key !== S.audioKey) {
    S.audioKey = key;
    const a = $("#audio");
    a.pause();
    if (key) { a.src = "/audio?k=" + encodeURIComponent(key); a.load(); } else a.removeAttribute("src");
    $("#b-audio").disabled = !key;
  }
  if (key !== S.bandsKey || !prev || prev.n !== c.n) loadBands(key);
  $("#t-dur").textContent = fmtTime(c.duration || 0, false);
  updateStats();
}

async function loadBands(key) {
  S.bandsKey = key;
  try { S.bands = (await (await fetch("/api/bands")).json()).bands; } catch { S.bands = null; }
  drawTimeline();
}

function applyStatus(st) {
  S.status = st;
  S.statusAt = performance.now();
  $("#t-play").textContent = st.playing ? "❚❚" : "▶︎";
  $("#t-play").classList.toggle("on", st.playing);
  if (!S.scrub) {
    $("#t-time").textContent = fmtTime(st.i / (st.fps || 30), true);
    $("#t-frame").textContent = `frame ${st.i + 1}/${st.n}`;
  }
  const msg = $("#stage-msg");
  if (st.warming) {
    msg.hidden = false;
    msg.textContent = `scaldo le scie… ${Math.round(100 * st.warming[0] / Math.max(1, st.warming[1]))}%`;
  } else msg.hidden = true;
  $("#cold").hidden = !st.cold || !!st.warming;
  drawTimeline();
  updateStats();
}

function updateStats() {
  const st = S.status, c = S.cfg;
  if (!st || !c) return;
  const parts = [];
  if (st.playing && st.fps_actual != null) {
    const ok = st.fps_actual >= st.fps * 0.92;
    parts.push(`<span class="${ok ? "good" : "bad"}"><b>${st.fps_actual.toFixed(1)}</b>/${st.fps} fps</span>`);
  } else parts.push(`${st.fps} fps`);
  parts.push(`deform ${st.ms.mask.toFixed(1)} · frame ${st.ms.step.toFixed(1)} · sfondo ${st.ms.bg.toFixed(1)} ms`);
  if (st.cache < 1) parts.push(`cache sfondo ${Math.round(st.cache * 100)}%`);
  parts.push(`preview <b>${c.W}×${c.H}</b> → video ${c.full?.[0]}×${c.full?.[1]}`);
  if (c.background) parts.push(`sfondo ${c.background.split("/").pop()} da ${(+c.bg_start).toFixed(1)}s`);
  if (c.audio) parts.push(`audio ${c.audio[0].split("/").pop()} da ${(+c.audio[1]).toFixed(1)}s`);
  $("#stats").innerHTML = parts.map((p) => `<span>${p}</span>`).join("");
}

function fmtTime(t, cents) {
  t = Math.max(0, t);
  const m = Math.floor(t / 60), s = t - m * 60;
  return `${m}:${(cents ? s.toFixed(2) : Math.floor(s).toString()).padStart(cents ? 5 : 2, "0")}`;
}

// ═════════════════════════════════════════════════════════════════════════════
//  Stage: video, A/B
// ═════════════════════════════════════════════════════════════════════════════
function layoutStage() {
  const wrap = $("#stage-wrap"), stage = $("#stage");
  const c = S.cfg;
  if (!c || !c.W) return;
  const pad = 20;
  const W = wrap.clientWidth - 2 * pad, H = wrap.clientHeight - 2 * pad;
  const k = Math.min(W / c.W, H / c.H);
  const w = Math.floor(c.W * k), h = Math.floor(c.H * k);
  Object.assign(stage.style, { width: w + "px", height: h + "px", left: (wrap.clientWidth - w) / 2 + "px", top: (wrap.clientHeight - h) / 2 + "px" });
  layoutSplit();
}

function layoutSplit() {
  const x = S.split * 100;
  $("#pin-layer").style.clipPath = `inset(0 ${100 - x}% 0 0)`;
  $("#divider").style.left = x + "%";
}

async function pinFrame() {
  const blob = await (await fetch("/snapshot.jpg?t=" + Date.now())).blob();
  if (S.pin) URL.revokeObjectURL(S.pin.url);
  const st = S.status;
  S.pin = { url: URL.createObjectURL(blob), i: st?.i ?? 0, preset: S.cfg?.preset };
  $("#pin").src = S.pin.url;
  $("#tag-a").textContent = `A · frame ${S.pin.i + 1}`;
  $("#b-compare").disabled = false;
  $("#b-pin").textContent = "📌 A aggiornato";
  setTimeout(() => ($("#b-pin").textContent = "📌 Fissa A"), 1200);
  if (!S.compare) setCompare(true);
}

function setCompare(on) {
  S.compare = on && !!S.pin;
  for (const id of ["#pin-layer", "#divider", "#tag-a", "#tag-b"]) $(id).hidden = !S.compare;
  $("#b-compare").classList.toggle("on", S.compare);
  layoutSplit();
}

function initDivider() {
  const div = $("#divider"), stage = $("#stage");
  let drag = false;
  const move = (e) => {
    const r = stage.getBoundingClientRect();
    S.split = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    layoutSplit();
  };
  div.addEventListener("pointerdown", (e) => { drag = true; div.setPointerCapture(e.pointerId); move(e); });
  div.addEventListener("pointermove", (e) => drag && move(e));
  div.addEventListener("pointerup", () => (drag = false));
  // clic sul video in modalità confronto: sposta la tendina lì
  stage.addEventListener("pointerdown", (e) => { if (S.compare && e.target !== div) move(e); });
}

// ─── il video: flusso MJPEG letto a mano e disegnato su canvas ──────────────
// (funziona in ogni browser e ogni frame porta il suo indice nell'header X-Frame)
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
async function streamLoop() {
  const cv = $("#live"), g = cv.getContext("2d");
  let latest = null, drawing = false;
  const draw = async () => {
    if (drawing || !latest) return;
    drawing = true;
    const { blob, i } = latest;
    latest = null;
    try {
      const bmp = await createImageBitmap(blob);
      if (cv.width !== bmp.width || cv.height !== bmp.height) { cv.width = bmp.width; cv.height = bmp.height; }
      g.drawImage(bmp, 0, 0);
      bmp.close();
      S.shownFrame = i;
    } catch { /* frame rovinato: salta */ }
    drawing = false;
    if (latest) draw();
  };
  const enc = new TextDecoder();
  for (;;) {
    try {
      const r = await fetch("/stream.mjpg?t=" + Date.now());
      const rd = r.body.getReader();
      let buf = new Uint8Array(0);
      const need = async (n) => {
        while (buf.length < n) {
          const { value, done } = await rd.read();
          if (done) throw new Error("fine");
          const nb = new Uint8Array(buf.length + value.length);
          nb.set(buf); nb.set(value, buf.length);
          buf = nb;
        }
      };
      const findHeaderEnd = () => {
        for (let k = 3; k < buf.length; k++) if (buf[k] === 10 && buf[k - 1] === 13 && buf[k - 2] === 10 && buf[k - 3] === 13) return k + 1;
        return -1;
      };
      for (;;) {
        let end;
        while ((end = findHeaderEnd()) < 0) await need(buf.length + 1);
        const head = enc.decode(buf.subarray(0, end));
        const len = parseInt(/Content-Length:\s*(\d+)/i.exec(head)?.[1] || "0", 10);
        const fi = parseInt(/X-Frame:\s*(\d+)/i.exec(head)?.[1] || "-1", 10);
        await need(end + len);
        latest = { blob: new Blob([buf.slice(end, end + len)], { type: "image/jpeg" }), i: fi };
        buf = buf.slice(end + len);
        draw();
      }
    } catch { /* connessione chiusa: riprova */ }
    await sleep(500);
  }
}

// ═════════════════════════════════════════════════════════════════════════════
//  Timeline con l'audio
// ═════════════════════════════════════════════════════════════════════════════
function drawTimeline() {
  const cv = $("#timeline");
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, w, h);
  g.fillStyle = "#0d0e12";
  g.fillRect(0, 0, w, h);
  const st = S.status, c = S.cfg;
  const n = st?.n || c?.n || 1;
  const dur = c?.duration || n / 30;
  const top = 14, bh = h - top - 4;

  // inviluppo audio: bassi / medi / alti
  if (S.bands) {
    const layers = [["high", "#6bd3ff"], ["mid", "#ffc46b"], ["bass", "#ff7a8a"]];
    for (const [k, col] of layers) {
      const arr = S.bands[k];
      if (!arr) continue;
      g.beginPath();
      g.moveTo(0, top + bh);
      for (let x = 0; x <= w; x += 1) {
        const v = arr[Math.min(arr.length - 1, Math.floor((x / w) * arr.length))] || 0;
        g.lineTo(x, top + bh - Math.sqrt(v) * bh * 0.95);
      }
      g.lineTo(w, top + bh);
      g.closePath();
      g.fillStyle = col + "38";
      g.fill();
      g.strokeStyle = col + "aa";
      g.lineWidth = 1;
      g.stroke();
    }
  }
  // cache dello sfondo
  if (st && st.cache < 1) {
    g.fillStyle = "#9f8cff55";
    g.fillRect(0, h - 2, w * st.cache, 2);
  }
  // tacche dei secondi
  g.fillStyle = "#5b6172";
  g.font = "10px ui-monospace, Menlo, monospace";
  const stepS = dur > 90 ? 10 : dur > 30 ? 5 : dur > 10 ? 2 : 1;
  for (let s = 0; s <= dur + 1e-6; s += stepS) {
    const x = (s / dur) * w;
    g.fillRect(Math.round(x), 0, 1, s % (stepS * 2) === 0 ? 6 : 4);
    if (s % (stepS * 2) === 0 && x < w - 20) g.fillText(fmtTime(s, false), x + 3, 10);
  }
  // A fissato
  if (S.pin && S.compare) {
    const x = ((S.pin.i + 0.5) / n) * w;
    g.fillStyle = "#ffffffaa";
    g.fillRect(x - 0.5, top, 1, bh);
    g.fillText("A", x + 3, top + 10);
  }
  // testina
  const i = S.scrub ? S.scrub.i : st?.i ?? 0;
  const x = ((i + 0.5) / n) * w;
  g.fillStyle = "#9f8cff";
  g.fillRect(x - 1, 0, 2, h);
  g.beginPath();
  g.moveTo(x - 5, 0); g.lineTo(x + 5, 0); g.lineTo(x, 6); g.closePath();
  g.fill();
}

function initTimeline() {
  const cv = $("#timeline"), hover = $("#tl-hover");
  const frameAt = (e) => {
    const r = cv.getBoundingClientRect();
    const n = S.status?.n || 1;
    return Math.max(0, Math.min(n - 1, Math.floor(((e.clientX - r.left) / r.width) * n)));
  };
  let last = 0;
  const seekFast = (i) => {
    const now = performance.now();
    if (now - last < 40) return;
    last = now;
    api("seek", { i, warm: false }, { quiet: true }).catch(() => {});
  };
  cv.addEventListener("pointerdown", (e) => {
    cv.setPointerCapture(e.pointerId);
    S.scrub = { i: frameAt(e) };
    seekFast(S.scrub.i);
    drawTimeline();
  });
  cv.addEventListener("pointermove", (e) => {
    const i = frameAt(e);
    const r = cv.getBoundingClientRect();
    hover.hidden = false;
    hover.style.left = (e.clientX - r.left) + "px";
    hover.textContent = `${fmtTime(i / (S.status?.fps || 30), true)} · ${i + 1}`;
    if (S.scrub) {
      S.scrub.i = i;
      $("#t-time").textContent = fmtTime(i / (S.status?.fps || 30), true);
      seekFast(i);
      drawTimeline();
    }
  });
  cv.addEventListener("pointerleave", () => (hover.hidden = true));
  const end = (e) => {
    if (!S.scrub) return;
    const i = frameAt(e);
    S.scrub = null;
    api("seek", { i, warm: true }).catch(() => {});
  };
  cv.addEventListener("pointerup", end);
  cv.addEventListener("pointercancel", () => (S.scrub = null));
}

// ═════════════════════════════════════════════════════════════════════════════
//  Audio sincronizzato col motore
// ═════════════════════════════════════════════════════════════════════════════
function audioLoop() {
  const a = $("#audio"), st = S.status;
  if (S.audioOn && st && st.playing && S.audioKey && !S.scrub) {
    const t = st.i / st.fps + (performance.now() - S.statusAt) / 1000;
    if (a.paused) {
      if (a.readyState >= 1) { a.currentTime = t; a.play().catch(() => {}); }
    } else if (Math.abs(a.currentTime - t) > 0.15) a.currentTime = t;
  } else if (!a.paused) a.pause();
  requestAnimationFrame(audioLoop);
}
function setAudio(on) {
  S.audioOn = on;
  store.set("audio", on);
  $("#b-audio").textContent = on ? "🔊" : "🔈";
  $("#b-audio").classList.toggle("on", on);
}

// ═════════════════════════════════════════════════════════════════════════════
//  Render, frame PNG
// ═════════════════════════════════════════════════════════════════════════════
function applyRender(r) {
  S.render = r;
  const bar = $("#jobbar");
  if (!r) { bar.hidden = true; return; }
  bar.hidden = false;
  bar.classList.toggle("done", r.state === "done");
  bar.classList.toggle("error", r.state === "error" || r.state === "cancelled");
  const what = r.kind === "frame" ? "Frame PNG a piena risoluzione" : r.test ? "Render di test" : "Render video";
  const pct = r.total ? r.done / r.total : 0;
  $("#job-fill").style.width = (r.state === "done" ? 100 : pct * 100) + "%";
  const acts = $("#job-actions");
  acts.innerHTML = "";
  const btn = (label, fn, cls = "") => { const b = document.createElement("button"); b.textContent = label; b.className = cls; b.onclick = fn; acts.append(b); };
  if (r.state === "running") {
    $("#job-text").textContent = r.total
      ? `${what} · ${r.done}/${r.total} · ${r.fps.toFixed(1)} fps · ETA ${r.eta}`
      : `${what} · preparo…`;
    btn("Annulla", () => api("render_cancel"), "ghost");
  } else if (r.state === "done") {
    $("#job-text").textContent = `✅ ${what}: ${r.path || ""}`;
    if (r.path) {
      btn(r.kind === "frame" ? "Guarda" : "Guarda", () => showMedia(r.path), "primary");
      btn("Apri", () => api("open", { path: r.path }));
      btn("Mostra nel Finder", () => api("open", { path: r.path, reveal: true }));
    }
    btn("✕", () => (bar.hidden = true), "ghost icon");
  } else {
    $("#job-text").textContent = r.state === "cancelled" ? `${what} annullato` : `❌ ${what} fallito: ${(r.log || "").split("\n").pop()}`;
    if (r.log) btn("Dettagli", () => toast(r.log, "error"), "ghost");
    btn("✕", () => (bar.hidden = true), "ghost icon");
  }
  $("#btn-render").disabled = $("#btn-test").disabled = $("#btn-frame").disabled = r.state === "running";
}

function showMedia(path) {
  const body = $("#media-body");
  body.innerHTML = "";
  const url = "/file?path=" + encodeURIComponent(path) + "&t=" + Date.now();
  if (/\.(png|jpe?g)$/i.test(path)) {
    const img = new Image();
    img.src = url;
    body.append(img);
  } else {
    const v = document.createElement("video");
    v.src = url; v.controls = true; v.autoplay = true; v.loop = true;
    body.append(v);
  }
  $("#media-title").textContent = path;
  $("#dlg-media").showModal();
}
$("#dlg-media").addEventListener("close", () => ($("#media-body").innerHTML = ""));

// ═════════════════════════════════════════════════════════════════════════════
//  Notifiche e dialoghi
// ═════════════════════════════════════════════════════════════════════════════
function toast(msg, kind = "info", ms) {
  const el = document.createElement("div");
  el.className = "toast " + kind;
  el.textContent = msg;
  $("#toasts").append(el);
  setTimeout(() => el.remove(), ms ?? (kind === "error" ? 7000 : 3000));
}

function confirmDlg(title, text, okLabel = "OK") {
  return new Promise((resolve) => {
    const d = $("#dlg-confirm");
    $("#confirm-title").textContent = title;
    $("#confirm-text").textContent = text;
    $("#confirm-ok").textContent = okLabel;
    d.returnValue = "";
    d.onclose = () => resolve(d.returnValue === "ok");
    d.showModal();
  });
}

async function saveAs() {
  const d = $("#dlg-saveas"), inp = $("#saveas-name");
  inp.value = S.cfg.editable ? S.cfg.preset + "_2" : S.cfg.preset;
  const check = () => ($("#saveas-warn").hidden = !(S.cfg.presets || []).includes(inp.value.trim()));
  inp.oninput = check;
  check();
  d.returnValue = "";
  d.showModal();
  inp.select();
  d.onclose = async () => {
    if (d.returnValue !== "ok") return;
    const name = inp.value.trim();
    await api("save", { name, overwrite: true }).catch(() => {});
  };
}

async function save() {
  if (!S.cfg.editable) return saveAs();
  if (!S.cfg.dirty) return toast("Niente da salvare");
  await api("save").catch(() => {});
}

// ═════════════════════════════════════════════════════════════════════════════
//  Eventi
// ═════════════════════════════════════════════════════════════════════════════
function connect() {
  const es = new EventSource("/events");
  es.addEventListener("config", (e) => applyConfig(JSON.parse(e.data)));
  es.addEventListener("status", (e) => applyStatus(JSON.parse(e.data)));
  es.addEventListener("render", (e) => applyRender(JSON.parse(e.data)));
  es.addEventListener("toast", (e) => { const d = JSON.parse(e.data); toast(d.msg, d.kind); });
  es.onopen = () => $("#stage-msg").textContent.startsWith("connessione") && ($("#stage-msg").hidden = true);
  es.onerror = () => {
    $("#stage-msg").hidden = false;
    $("#stage-msg").textContent = "connessione persa — lo studio è ancora aperto nel terminale?";
  };
}

function bind() {
  $("#t-play").onclick = () => api("toggle");
  $("#t-start").onclick = () => api("seek", { i: 0, warm: false });
  $("#t-prev").onclick = () => api("step", { d: -1 });
  $("#t-next").onclick = () => api("step", { d: 1 });
  $("#b-pin").onclick = pinFrame;
  $("#b-compare").onclick = () => setCompare(!S.compare);
  $("#b-audio").onclick = () => setAudio(!S.audioOn);
  $("#undo").onclick = undo;
  $("#redo").onclick = redo;
  $("#save").onclick = save;
  $("#saveas").onclick = saveAs;
  $("#revert").onclick = async () => {
    if (await confirmDlg("Ripristinare il preset?", "Le modifiche non salvate vanno perse (puoi sempre fare ⌘Z).", "Ripristina")) {
      beginEdit(); await api("revert").catch(() => {}); endEdit();
    }
  };
  $("#preset").onchange = async (e) => {
    const name = e.target.value;
    if (S.cfg.dirty && !(await confirmDlg(`Aprire «${name}»?`, `Le modifiche non salvate a «${S.cfg.preset}» vanno perse.`, "Apri"))) {
      e.target.value = S.cfg.preset;
      return;
    }
    S.undo = []; S.redo = []; updateUndoButtons();
    await api("load", { preset: name }).catch(() => (e.target.value = S.cfg.preset));
  };
  $("#seed").onchange = (e) => api("seed", { action: "set", value: parseInt(e.target.value || "0", 10) }).catch(() => {});
  $("#seed").onkeydown = (e) => { if (e.key === "Enter") e.target.blur(); e.stopPropagation(); };
  $("#seed-rnd").onclick = () => { beginEdit(); api("seed", { action: "random" }).catch(() => {}).finally(endEdit); };
  $("#seed-fix").onclick = () => { beginEdit(); api("seed", { action: S.cfg.seed_fixed ? "unfix" : "fix" }).catch(() => {}).finally(endEdit); };
  $("#view-scale").onchange = (e) => api("view", { scale: parseFloat(e.target.value) });
  $("#view-fps").onchange = (e) => api("view", { fps: e.target.value ? parseInt(e.target.value, 10) : null });
  $("#btn-render").onclick = () => api("render", { test: false }).catch(() => {});
  $("#btn-test").onclick = () => api("render", { test: true }).catch(() => {});
  $("#btn-frame").onclick = () => api("frame").catch(() => {});
  $("#btn-help").onclick = () => $("#dlg-help").showModal();
  $("#search").oninput = applyFilter;
  $("#search").onkeydown = (e) => { if (e.key === "Escape") { e.target.value = ""; applyFilter(); e.target.blur(); } e.stopPropagation(); };
  $("#only-set").onchange = applyFilter;
  $("#fold").onclick = () => {
    const dets = $$("details.grp:not(.hidden)");
    const open = !dets.every((d) => d.open);
    dets.forEach((d) => (d.open = open));
  };

  document.addEventListener("keydown", (e) => {
    const mod = e.metaKey || e.ctrlKey;
    if (mod && e.key.toLowerCase() === "s") { e.preventDefault(); save(); return; }
    const typing = ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)
      && document.activeElement.type !== "range" && document.activeElement.type !== "checkbox";
    if (typing || $("dialog[open]")) return;
    if (mod && e.key.toLowerCase() === "z") { e.preventDefault(); e.shiftKey ? redo() : undo(); return; }
    if (mod) return;
    const fps = S.status?.fps || 30, i = S.status?.i || 0;
    switch (e.key) {
      case " ": e.preventDefault(); api("toggle"); break;
      case "ArrowRight": e.preventDefault(); e.shiftKey ? api("seek", { i: i + fps, warm: true }) : api("step", { d: 1 }); break;
      case "ArrowLeft": e.preventDefault(); e.shiftKey ? api("seek", { i: i - fps, warm: true }) : api("step", { d: -1 }); break;
      case "Home": e.preventDefault(); api("seek", { i: 0, warm: false }); break;
      case "p": case "P": pinFrame(); break;
      case "c": case "C": setCompare(!S.compare); break;
      case "m": case "M": setAudio(!S.audioOn); break;
      case "/": e.preventDefault(); $("#search").focus(); break;
      case "?": $("#dlg-help").showModal(); break;
    }
  });
  new ResizeObserver(() => { layoutStage(); drawTimeline(); }).observe($("#stage-wrap"));
  new ResizeObserver(drawTimeline).observe($("#timeline"));
}

async function main() {
  bind();
  initDivider();
  initTimeline();
  setAudio(S.audioOn);
  updateUndoButtons();
  const [schema, state] = await Promise.all([
    fetch("/api/schema").then((r) => r.json()),
    fetch("/api/state").then((r) => r.json()),
  ]);
  S.schema = schema;
  S.files = state.files;
  for (const [k, list] of Object.entries(S.files)) {
    const dl = $("#dl-" + k);
    if (dl) for (const f of list) dl.append(new Option(f, f));
  }
  buildPanel();
  applyConfig(state.config);
  applyStatus(state.status);
  applyRender(state.render);
  connect();
  streamLoop();
  requestAnimationFrame(audioLoop);
}
main();
