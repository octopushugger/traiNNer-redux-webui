'use strict';

// ── Constants ─────────────────────────────────────────────────────────────────
const TT           = 22;   // tab thickness (px)
const TL           = 148;  // tab length (px)
const HT           = 5;    // handle thickness (px)
const MIN_CENTER_W = 200;
const MIN_UPPER_H  = TL + 2 * 50;  // 248
const ANIM_MS      = 220;
const DRAG_THRESH  = 4;
const STORAGE_KEY  = 'traiNNer_layout';

const BG     = '#111827';
const SURF   = '#1f2937';
const SURF2  = '#2d3748';
const BORDER = '#374151';
const ACC    = '#6366f1';
const TEXT   = '#f9fafb';
const MUTED  = '#9ca3af';
const GREEN  = '#10b981';
const RED    = '#ef4444';

// ── Layout persistence ────────────────────────────────────────────────────────
function defaultLw() { return 300; }
function defaultRw() { return Math.round(window.innerWidth  * 0.33); }
function defaultBh() { return Math.round(window.innerHeight * 0.33); }

function saveLayout() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ lw: S._lw, rw: S._rw, bh: S._bh }));
  } catch (_) {}
}

function loadLayout() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null');
    S._lw = saved?.lw ?? defaultLw();
    S._rw = saved?.rw ?? defaultRw();
    S._bh = saved?.bh ?? defaultBh();
  } catch (_) {
    S._lw = defaultLw(); S._rw = defaultRw(); S._bh = defaultBh();
  }
}

// ── App state ─────────────────────────────────────────────────────────────────
const S = {
  _lw: 0, _rw: 0, _bh: 0,   // overwritten by loadLayout() in init()
  selectedKey: null,
  trainingKey:    null,   // key of the experiment currently training, or null
  trainingName:   null,   // display name of training/last-trained experiment
  trainingStatus: null,   // last status string,persists until next Start
  ws: null,
  experiments: [],
  // Iteration extrapolation
  lastIter:     0,
  lastIterTime: 0,   // Date.now() when lastIter was received
  lastIts:      0,
  // Per-experiment persistent stats
  statsCache:    {},   // key → {epoch, iter, its, eta, lr_val, vram}
  statsDebounce: null, // timer for debounced server save
  // Console line buffer (CR-aware)
  consoleHTML:       '',   // committed lines as pre-rendered HTML
  consoleCurrentRaw: '',   // current partial line (raw, overwritten by \r)
  // Config dirty tracking
  originalConfig: null,
  configDirty:    false,
  // Validation popup
  validating: false,
  // Server reachability
  configLoadFailed: false,
  // Visualization images
  vizChecked: new Set(),   // keys checked for viz images this session
  vizData:    {},          // key → {folders:[{name,images:[]}]}
};

// ── Panel dimension setters with push-through logic ───────────────────────────
function setLw(v) {
  S._lw = Math.max(0, Math.round(v));
  const max = Math.max(0, root().clientWidth - MIN_CENTER_W);
  if (S._lw + S._rw > max) {
    S._rw = Math.max(0, max - S._lw);
    S._lw = Math.min(S._lw, max);   // clamp self if opposite panel already at 0
  }
}

function setRw(v) {
  S._rw = Math.max(0, Math.round(v));
  const max = Math.max(0, root().clientWidth - MIN_CENTER_W);
  if (S._lw + S._rw > max) {
    S._lw = Math.max(0, max - S._rw);
    S._rw = Math.min(S._rw, max);   // clamp self if opposite panel already at 0
  }
}

function setBh(v) {
  const H = root().clientHeight;
  S._bh = Math.max(0, Math.min(Math.round(v), H - MIN_UPPER_H));
}

// ── Element cache ─────────────────────────────────────────────────────────────
const _cache = {};
function $(id) { return _cache[id] || (_cache[id] = document.getElementById(id)); }
function root() { return $('root'); }

// ── Tab hover state ───────────────────────────────────────────────────────────
const tabHover = { lw: false, rw: false, bh: false };

// ── Layout ────────────────────────────────────────────────────────────────────
function setGeom(el, x, y, w, h) {
  el.style.left   = x + 'px';
  el.style.top    = y + 'px';
  el.style.width  = w + 'px';
  el.style.height = h + 'px';
}

function relayout() {
  const W  = root().clientWidth;
  const H  = root().clientHeight;
  const lw = S._lw, rw = S._rw, bh = S._bh;
  const upperH  = H - bh;
  const centerW = W - lw - rw;
  const tabY    = Math.max(0, Math.floor((upperH - TL) / 2));

  setGeom($('left-panel'),   0,      0,      lw,      upperH);
  setGeom($('center'),       lw,     0,      centerW, upperH);
  setGeom($('right-panel'),  W - rw, 0,      rw,      H);      // full height
  setGeom($('bottom-panel'), 0,      upperH, W - rw,  bh);     // shrinks with right panel

  // Tabs
  setGeom($('tab-left'),   lw,                              tabY,        TT, TL);
  setGeom($('tab-right'),  W - rw - TT,                    tabY,        TT, TL);
  setGeom($('tab-bottom'), lw + Math.floor((centerW-TL)/2), upperH - TT, TL, TT);

  // Handles sit at the panel edge; tabs (z-index 10) overlap them in the tab region
  const hl = $('handle-left');
  setGeom(hl, lw, 0, HT, upperH);
  hl.style.display = lw > 0 ? 'block' : 'none';

  const hr = $('handle-right');
  setGeom(hr, W - rw - HT, 0, HT, upperH);
  hr.style.display = rw > 0 ? 'block' : 'none';

  const hb = $('handle-bottom');
  setGeom(hb, 0, upperH - HT, W - rw, HT);  // stop at right panel edge
  hb.style.display = bh > 0 ? 'block' : 'none';

  updateConfigDirtyUI();

  // Blurred tab shadows — match each tab's exact position/size
  const shadowL = document.getElementById('tab-shadow-left');
  const shadowR = document.getElementById('tab-shadow-right');
  const shadowB = document.getElementById('tab-shadow-bottom');
  const SI = 7; // inset so shadow rect is slightly smaller than the tab
  if (shadowL) { shadowL.style.display = lw > 0 ? 'block' : 'none'; setGeom(shadowL, lw           + SI, tabY        + SI, TT - SI * 2, TL - SI * 2); }
  if (shadowR) { shadowR.style.display = rw > 0 ? 'block' : 'none'; setGeom(shadowR, W - rw - TT  + SI, tabY        + SI, TT - SI * 2, TL - SI * 2); }
  if (shadowB) { shadowB.style.display = bh > 0 ? 'block' : 'none'; setGeom(shadowB, lw + Math.floor((centerW - TL) / 2) + SI, upperH - TT + SI, TL - SI * 2, TT - SI * 2); }

  drawTabs();
  redrawAllCharts();
}

// ── Chrome tab shape (Canvas) ─────────────────────────────────────────────────
function drawChromeShape(ctx, W, H, fill, label) {
  const S = 10;
  if (W < S * 3.2 + 4 || H < 4) return;
  const sl = Math.min(S * 1.6, W / 2 - 2);
  const sr = W - sl;

  ctx.beginPath();
  ctx.moveTo(sl, 0);
  ctx.lineTo(sr, 0);
  ctx.bezierCurveTo(sr + S, 0,  W,     H - S, W, H);
  ctx.lineTo(0, H);
  ctx.bezierCurveTo(0,     H - S, sl - S, 0,     sl, 0);
  ctx.closePath();

  ctx.fillStyle = fill;
  ctx.fill();

  ctx.fillStyle = TEXT;
  ctx.font = '500 11px -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, W / 2, H / 2 + 0.5);
}

function drawTabLeft() {
  const canvas = $('tab-left');
  canvas.width = TT; canvas.height = TL;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, TT, TL);
  const fill = tabHover.lw ? SURF2 : SURF;
  // Rotate 90° CW so the Chrome shape sits vertically; flat edge faces left panel
  ctx.save();
  ctx.translate(TT, 0);
  ctx.rotate(Math.PI / 2);
  drawChromeShape(ctx, TL, TT, fill, 'EXPERIMENTS');
  ctx.restore();
}

function drawTabRight() {
  const canvas = $('tab-right');
  canvas.width = TT; canvas.height = TL;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, TT, TL);
  const fill = tabHover.rw ? SURF2 : SURF;
  // Rotate 90° CCW; flat edge faces right panel
  ctx.save();
  ctx.translate(0, TL);
  ctx.rotate(-Math.PI / 2);
  drawChromeShape(ctx, TL, TT, fill, 'CONFIG');
  ctx.restore();
}

function drawTabBottom() {
  const canvas = $('tab-bottom');
  canvas.width = TL; canvas.height = TT;
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, TL, TT);
  const fill = tabHover.bh ? '#1a1a1a' : '#000000';
  drawChromeShape(ctx, TL, TT, fill, 'CONSOLE OUTPUT');
}

function drawTabs() { drawTabLeft(); drawTabRight(); drawTabBottom(); }

// ── Animation ─────────────────────────────────────────────────────────────────
const anims = {};

function easeInOut(t) { return t < 0.5 ? 2*t*t : -1 + (4 - 2*t)*t; }

function animateTo(key, toValue, ms) {
  ms = ms || ANIM_MS;
  if (anims[key]) cancelAnimationFrame(anims[key]);
  const from  = S['_' + key];
  const start = performance.now();

  const setFn = { lw: setLw, rw: setRw, bh: setBh }[key];

  function frame(now) {
    const t = Math.min((now - start) / ms, 1);
    setFn(from + (toValue - from) * easeInOut(t));
    relayout();
    if (t < 1) anims[key] = requestAnimationFrame(frame);
    else      { delete anims[key]; saveLayout(); }
  }
  anims[key] = requestAnimationFrame(frame);
}

// ── Tab interactions (drag-to-resize + click-to-toggle) ───────────────────────
function setupTab(tabId, key, initOpenVal) {
  const canvas  = $(tabId);
  const isHoriz = (key !== 'bh');
  let openVal   = initOpenVal;   // mutable,updated to last non-zero size before each close

  let pressed = false, isDrag = false, startPos = 0, lastPos = 0;
  const getPos = (e) => isHoriz ? e.clientX : e.clientY;

  canvas.addEventListener('mouseenter', () => { tabHover[key] = true;  drawTabs(); });
  canvas.addEventListener('mouseleave', () => { if (!pressed) { tabHover[key] = false; drawTabs(); } });

  canvas.addEventListener('mousedown', (e) => {
    e.preventDefault();
    pressed  = true;
    isDrag   = false;
    startPos = lastPos = getPos(e);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });

  function onMove(e) {
    if (!pressed) return;
    const pos = getPos(e);
    if (Math.abs(pos - startPos) >= DRAG_THRESH) isDrag = true;
    if (isDrag) {
      const delta = pos - lastPos;
      lastPos = pos;
      if      (key === 'lw') setLw(S._lw + delta);
      else if (key === 'rw') setRw(S._rw - delta);
      else                   setBh(S._bh - delta);
      relayout();
    }
  }

  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup',   onUp);
    const wasDrag = isDrag;
    pressed = false; isDrag = false;
    tabHover[key] = false;
    drawTabs();
    if (wasDrag) {
      // Remember the dragged-to size for next reopen
      if (S['_' + key] > 0) openVal = S['_' + key];
      saveLayout();
    } else {
      if (anims[key]) cancelAnimationFrame(anims[key]);
      if (S['_' + key] > 0) {
        openVal = S['_' + key];   // snapshot current size before closing
        animateTo(key, 0);
      } else {
        animateTo(key, openVal);  // reopen to last known size
      }
      // animation end saves via saveLayout() inside animateTo → frame
    }
  }
}

// ── Resize handle interactions ─────────────────────────────────────────────────
function setupHandle(handleId, key) {
  const el     = $(handleId);
  const isHoriz = (key !== 'bh');
  let dragging = false, last = 0;
  const getPos = (e) => isHoriz ? e.clientX : e.clientY;

  el.addEventListener('mousedown', (e) => {
    e.preventDefault();
    dragging = true;
    last = getPos(e);
    el.classList.add('dragging');
    document.body.style.cursor = isHoriz ? 'ew-resize' : 'ns-resize';
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onUp);
  });

  function onMove(e) {
    if (!dragging) return;
    const pos   = getPos(e);
    const delta = pos - last;
    last = pos;
    if      (key === 'lw') setLw(S._lw + delta);
    else if (key === 'rw') setRw(S._rw - delta);
    else                   setBh(S._bh - delta);
    relayout();
  }

  function onUp() {
    document.removeEventListener('mousemove', onMove);
    document.removeEventListener('mouseup',   onUp);
    dragging = false;
    el.classList.remove('dragging');
    document.body.style.cursor = '';
    saveLayout();
  }
}

// ── Server reachability ───────────────────────────────────────────────────────

async function apiFetch(url, opts) {
  try {
    const res = await fetch(url, opts);
    setServerReachable(true);
    return res;
  } catch (_) {
    setServerReachable(false);
    return null;
  }
}

function setServerReachable(ok) {
  const banner  = document.getElementById('server-banner');
  const wasDown = !banner.classList.contains('hidden');
  banner.classList.toggle('hidden', ok);
  // When the server comes back, re-load the selected experiment's config if
  // the previous attempt failed.
  if (ok && wasDown && S.selectedKey && S.configLoadFailed) {
    S.configLoadFailed = false;
    selectExperiment(S.selectedKey);
  }
}

// ── Experiments ───────────────────────────────────────────────────────────────
async function loadExperiments() {
  const res = await apiFetch('/api/experiments');
  if (!res) return;
  const data = await res.json();
  S.experiments = data.experiments;
  // Seed stats cache from persisted server data
  for (const exp of S.experiments) {
    if (exp.stats) S.statsCache[exp.key] = exp.stats;
  }
  renderExperiments();
  await restoreTrainingState();
}

async function restoreTrainingState() {
  const res = await apiFetch('/api/training');
  if (!res) return;
  const data = await res.json();
  if (!data.key) return;
  const exp = S.experiments.find(e => e.key === data.key);
  S.trainingKey  = data.key;
  S.trainingName = exp ? exp.name : data.key;
  connectWs(data.key);
  setTrainingStatus('Training');
  updateTrainBtn();
}

function fmtDate(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: 'short', day: 'numeric', year: 'numeric' });
}

const SORT_KEY = 'traiNNer_exp_sort';

function getSortOrder() {
  return localStorage.getItem(SORT_KEY) || 'last_trained';
}

function sortedExperiments() {
  const order = getSortOrder();
  const exps  = [...S.experiments];
  exps.sort((a, b) => {
    switch (order) {
      case 'newest':
        return (b.created_at || '') > (a.created_at || '') ? 1 : -1;
      case 'oldest':
        return (a.created_at || '') > (b.created_at || '') ? 1 : -1;
      case 'name_asc':
        return a.name.localeCompare(b.name);
      case 'name_desc':
        return b.name.localeCompare(a.name);
      default: // last_trained: most recently run first, never-run fall to bottom then by created desc
        if (a.last_run_at && b.last_run_at)
          return b.last_run_at > a.last_run_at ? 1 : -1;
        if (a.last_run_at) return -1;
        if (b.last_run_at) return  1;
        return (b.created_at || '') > (a.created_at || '') ? 1 : -1;
    }
  });
  return exps;
}

function renderExperiments() {
  const list = $('experiments-list');
  list.innerHTML = '';
  for (const exp of sortedExperiments()) {
    const div = document.createElement('div');
    div.className   = 'experiment-item' + (exp.key === S.selectedKey ? ' active' : '');
    div.title       = exp.name + ' · ' + exp.arch;
    div.dataset.key = exp.key;

    const mainDiv = document.createElement('div');
    mainDiv.className = 'exp-main';

    const nameSpan = document.createElement('span');
    nameSpan.className   = 'exp-name';
    nameSpan.textContent = exp.name;

    const datesSpan = document.createElement('span');
    datesSpan.className = 'exp-dates';
    const created     = fmtDate(exp.created_at);
    const lastTrained = fmtDate(exp.last_run_at);
    datesSpan.textContent = (created ? 'Created ' + created : '')
      + (created && lastTrained ? ' · ' : '')
      + (lastTrained ? 'Trained ' + lastTrained : '');

    const archSpan = document.createElement('span');
    archSpan.className   = 'exp-arch';
    archSpan.textContent = exp.arch;

    mainDiv.appendChild(nameSpan);
    if (datesSpan.textContent) mainDiv.appendChild(datesSpan);
    div.appendChild(mainDiv);
    div.appendChild(archSpan);
    div.addEventListener('click', () => selectExperiment(exp.key));
    div.addEventListener('contextmenu', (e) => { e.preventDefault(); showCtxMenu(e.clientX, e.clientY, exp.key); });
    list.appendChild(div);
  }
}

// ── Per-experiment stats display ──────────────────────────────────────────────
function renderStats(key) {
  const s = key ? S.statsCache[key] : null;
  $('stat-epoch').textContent = s?.epoch != null ? s.epoch.toLocaleString() : '—';
  $('stat-iter').textContent  = s?.iter  != null ? s.iter.toLocaleString()  : '—';
  $('stat-iter-total').textContent = s?.totalIters ? '/ ' + s.totalIters.toLocaleString() : '';
  $('stat-its').textContent   = s?.its   != null ? s.its.toFixed(2)         : '—';
  $('stat-eta').textContent   = s?.eta   != null ? s.eta                    : '—';
  if (s?.lr_val != null) {
    $('stat-lr').textContent       = s.lr_val.toExponential(2);
    const lrNormal = s.lr_val < 1e-7 ? s.lr_val.toExponential(2) : String(s.lr_val);
    $('stat-lr-normal').textContent = '(' + lrNormal + ')';
  } else {
    $('stat-lr').textContent        = '—';
    $('stat-lr-normal').textContent = '';
  }
  $('stat-vram').textContent = s?.vram != null ? s.vram.toFixed(2) + ' GB' : '—';
}

async function saveStatsToServer(key) {
  const stats = S.statsCache[key];
  if (!stats) return;
  await fetch('/api/experiments/' + encodeURIComponent(key) + '/stats', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(stats),
  });
}

async function selectExperiment(key) {
  S.selectedKey = key;
  renderExperiments();
  renderStats(key);
  loadGraphs(key);
  checkVisualization(key);

  // Load YAML config
  $('config-editor').value = '';
  updateHighlight();
  document.getElementById('config-loading').classList.remove('hidden');
  const res = await apiFetch('/api/experiments/' + encodeURIComponent(key) + '/config');
  document.getElementById('config-loading').classList.add('hidden');
  if (!res || !res.ok) {
    S.originalConfig = '';
    S.configLoadFailed = true;
    setConfigDirty(false);
    return;
  }
  S.configLoadFailed = false;
  const data = await res.json();
  $('config-editor').value = data.content;
  S.originalConfig = data.content;
  setConfigDirty(false);
  updateHighlight();

  // Parse total_iter and print_freq from config
  const totalIters = parseTotalIter(data.content);
  const printFreq  = parsePrintFreq(data.content);
  if (!S.statsCache[key]) S.statsCache[key] = {};
  S.statsCache[key].totalIters = totalIters;
  S.statsCache[key].printFreq  = printFreq;
  $('stat-iter-total').textContent = totalIters ? '/ ' + totalIters.toLocaleString() : '';

  updateTrainBtn();

  // Open right panel if closed
  if (S._rw === 0) animateTo('rw', defaultRw());
}

// ── Context menu ──────────────────────────────────────────────────────────────
let ctxKey = null;

function showCtxMenu(x, y, key) {
  ctxKey = key;
  const m = $('ctx-menu');
  m.classList.remove('hidden');
  // Keep menu inside viewport
  const vw = window.innerWidth, vh = window.innerHeight;
  const mw = 170, mh = 130;
  m.style.left = Math.min(x, vw - mw) + 'px';
  m.style.top  = Math.min(y, vh - mh) + 'px';
}

document.addEventListener('click', () => {
  $('ctx-menu').classList.add('hidden');
  ctxKey = null;
});

// ── New experiment modal ───────────────────────────────────────────────────────
async function showNewExperimentModal() {
  const res  = await fetch('/api/archs');
  const data = await res.json();
  const sel  = $('modal-arch');
  sel.innerHTML = '';
  for (const arch of data.archs) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = arch;
    sel.appendChild(opt);
  }
  if (data.archs.length) await loadTemplates(sel.value);
  $('modal-overlay').classList.remove('hidden');
  $('modal-name').focus();
}

async function loadTemplates(arch) {
  if (!arch) return;
  const res  = await fetch('/api/archs/' + encodeURIComponent(arch) + '/templates');
  const data = await res.json();
  const sel  = $('modal-template');
  sel.innerHTML = '';
  for (const t of data.templates) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = t;
    sel.appendChild(opt);
  }
  if (data.templates.length) $('modal-name').value = data.templates[0];
}

function closeModal() { $('modal-overlay').classList.add('hidden'); }

async function createExperiment() {
  const arch = $('modal-arch').value;
  const name = $('modal-name').value.trim();
  if (!name) { $('modal-name').focus(); return; }
  const res = await fetch('/api/experiments', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ arch, name }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err.detail || 'Failed to create experiment');
    return;
  }
  const exp = await res.json();
  closeModal();
  await loadExperiments();
  selectExperiment(exp.key);
}

// ── Rename experiment modal ────────────────────────────────────────────────────
let renameKey = null;

function showRenameModal(key) {
  renameKey = key;
  const exp = S.experiments.find(e => e.key === key);
  $('rename-input').value = exp ? exp.name : '';
  $('rename-overlay').classList.remove('hidden');
  $('rename-input').select();
  $('rename-input').focus();
}

function closeRenameModal() {
  $('rename-overlay').classList.add('hidden');
  renameKey = null;
}

async function submitRename() {
  if (!renameKey) return;
  const newName = $('rename-input').value.trim();
  if (!newName) { $('rename-input').focus(); return; }
  const res = await fetch('/api/experiments/' + encodeURIComponent(renameKey) + '/rename', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ new_name: newName }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    alert(err.detail || 'Failed to rename experiment');
    return;
  }
  const updated = await res.json();
  const wasSelected = S.selectedKey === renameKey;
  if (wasSelected) S.selectedKey = updated.key;
  closeRenameModal();
  await loadExperiments();
  if (wasSelected) selectExperiment(updated.key);
}

// ── Config editor ─────────────────────────────────────────────────────────────
function updateConfigDirtyUI() {
  const W   = root().clientWidth;
  const rw  = S._rw;
  const dirty = S.configDirty;

  // Position floating buttons centered over the right panel, clamped to its edges
  const btns = document.getElementById('config-save-btns');
  if (btns) {
    const btnW      = btns.offsetWidth || 0;
    const panelLeft = W - rw;
    const idealLeft = W - rw / 2;
    // Only clamp left edge — let buttons slide off-screen to the right as the panel narrows
    const clamped   = Math.max(idealLeft, panelLeft + btnW / 2 + 8);
    btns.style.left = clamped + 'px';
  }

  // When the panel is fully closed, show a toast instead
  const toast = document.getElementById('config-dirty-toast');
  if (toast) toast.classList.toggle('visible', dirty && rw === 0);
}

function setConfigDirty(dirty) {
  S.configDirty = dirty;
  $('config-save-btns').classList.toggle('visible', dirty && S._rw > 0);
  updateConfigDirtyUI();
}

function discardConfig() {
  if (S.originalConfig === null) return;
  if (!confirm('Discard all unsaved changes to this config?')) return;
  $('config-editor').value = S.originalConfig;
  updateHighlight();
  setConfigDirty(false);
}

async function saveConfig() {
  if (!S.selectedKey) return;
  const content = $('config-editor').value;
  await fetch('/api/experiments/' + encodeURIComponent(S.selectedKey) + '/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });

  S.originalConfig = content;

  // Re-parse total_iter and print_freq in case they changed
  const totalIters = parseTotalIter(content);
  const printFreq  = parsePrintFreq(content);
  if (!S.statsCache[S.selectedKey]) S.statsCache[S.selectedKey] = {};
  S.statsCache[S.selectedKey].totalIters = totalIters;
  S.statsCache[S.selectedKey].printFreq  = printFreq;
  $('stat-iter-total').textContent = totalIters ? '/ ' + totalIters.toLocaleString() : '';

  setConfigDirty(false);
}

// ── Training ──────────────────────────────────────────────────────────────────
function updateTrainBtn() {
  const btn = $('train-btn'), txt = $('train-btn-text'), sc = $('train-btn-shortcut');
  const isTrainingThis  = S.trainingKey !== null && S.trainingKey === S.selectedKey;
  const isTrainingOther = S.trainingKey !== null && S.trainingKey !== S.selectedKey;

  if (isTrainingThis) {
    btn.disabled = false;
    btn.classList.add('running');
    txt.textContent = '■ Stop Training';
    sc.textContent  = '(Ctrl+Alt+Enter)';
  } else if (isTrainingOther) {
    btn.disabled = true;
    btn.classList.remove('running');
    txt.textContent = '▶ Start Training';
    sc.textContent  = '(Ctrl+Enter)';
  } else {
    btn.disabled = !S.selectedKey;
    btn.classList.remove('running');
    txt.textContent = '▶ Start Training';
    sc.textContent  = '(Ctrl+Enter)';
  }
  renderTrainingStatus();   // re-render so subtext reflects current selectedKey
}

async function toggleTraining() {
  if (!S.selectedKey) return;
  if (S.trainingKey === S.selectedKey) {
    // Stop the running experiment — kill extrapolation and progress bar immediately
    S.lastIterTime = 0;
    clearIterProgressBar();
    hideValidationPopup();
    const res = await fetch('/api/experiments/' + encodeURIComponent(S.selectedKey) + '/stop', { method: 'POST' });
    if (!res.ok) {
      S.trainingKey = null; updateTrainBtn(); setTrainingStatus(null);
    } else {
      setTrainingStatus('Saving Model');  // optimistic,SIGINT triggers a checkpoint save
    }
    // Sentinel from _stream_output will update the UI once the process exits
  } else if (S.trainingKey === null) {
    // Start training,clear any previous status and record the experiment name
    const startExp = S.experiments.find(e => e.key === S.selectedKey);
    S.trainingName   = startExp ? startExp.name : S.selectedKey;
    S.trainingStatus = null;
    S.trainingKey    = S.selectedKey;
    S.lastIter = 0; S.lastIterTime = 0; S.lastIts = 0;
    connectWs(S.selectedKey);
    updateTrainBtn();
    setTrainingStatus('Loading');
    const res = await fetch('/api/experiments/' + encodeURIComponent(S.selectedKey) + '/start', { method: 'POST' });
    if (!res.ok) {
      S.trainingKey = null;
      const e = await res.json().catch(() => ({}));
      alert(e.detail || 'Failed to start');
      updateTrainBtn();
      setTrainingStatus(null);
    } else {
      // Optimistically update the in-memory last_run_at so the card reflects
      // today's date immediately without waiting for a full loadExperiments().
      if (startExp) { startExp.last_run_at = new Date().toISOString(); renderExperiments(); }
    }
  }
}

// ── ANSI → HTML ───────────────────────────────────────────────────────────────
// Standard 8 + 8 bright colors
const _ANSI_FG = [
  '#4a4a4a','#ef4444','#22c55e','#f59e0b',  // 0 black, 1 red, 2 green, 3 yellow
  '#3b82f6','#a855f7','#06b6d4','#d1d5db',  // 4 blue, 5 magenta, 6 cyan, 7 white
  '#6b7280','#f87171','#86efac','#fde68a',  // 8-11 bright
  '#93c5fd','#d8b4fe','#67e8f9','#f9fafb',  // 12-15 bright
];

function _ansi256(n) {
  if (n < 8)  return _ANSI_FG[n];
  if (n < 16) return _ANSI_FG[n];
  if (n < 232) {
    n -= 16;
    const b = n % 6, g = Math.floor(n/6) % 6, r = Math.floor(n/36);
    const v = x => x ? x*40+55 : 0;
    return `rgb(${v(r)},${v(g)},${v(b)})`;
  }
  const l = (n - 232) * 10 + 8;
  return `rgb(${l},${l},${l})`;
}

function ansiToHtml(raw) {
  // Strip non-SGR control sequences: cursor movement, erase, screen control, OSC, etc.
  // We preserve ESC[...m (SGR) which carry color/style — those are handled below.
  const clean = raw
    .replace(/\x1b\[[0-9;]*[A-LN-Za-ln-z]/g, '') // CSI non-SGR (cursor up/down, erase line — excludes 'm' which is SGR)
    .replace(/\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)/g, '')  // OSC (window title, hyperlinks…)
    .replace(/\x1b[@-Z\\-_]/g, '');              // other 2-char escape sequences

  // Handle \r: within each \n-delimited line, keep only text after the last \r
  const crStripped = clean.replace(/[^\n]*\r(?!\n)/g, '').replace(/\r\n/g, '\n');

  // Escape HTML entities before injecting
  let text = crStripped.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');

  let out = '', openSpans = 0;
  let bold = false, fg = null;

  // Split on SGR sequences (ESC [ … m)
  const parts = text.split(/(\x1b\[[0-9;]*m)/);

  for (const part of parts) {
    if (!part.startsWith('\x1b[')) { out += part; continue; }

    // Close existing style spans before applying new ones
    while (openSpans-- > 0) out += '</span>';
    openSpans = 0;
    bold = false; fg = null;

    const seq = part.slice(2, -1);
    const codes = seq === '' ? [0] : seq.split(';').map(Number);

    for (let i = 0; i < codes.length; i++) {
      const c = codes[i];
      if      (c === 0)                   { bold = false; fg = null; }
      else if (c === 1)                   { bold = true; }
      else if (c >= 30 && c <= 37)        { fg = _ANSI_FG[c - 30]; }
      else if (c >= 90 && c <= 97)        { fg = _ANSI_FG[c - 90 + 8]; }
      else if (c === 39)                  { fg = null; }
      else if (c === 38 && codes[i+1] === 5 && codes[i+2] !== undefined) {
        fg = _ansi256(codes[i+2]); i += 2;
      } else if (c === 38 && codes[i+1] === 2) {
        fg = `rgb(${codes[i+2]},${codes[i+3]},${codes[i+4]})`; i += 4;
      }
    }

    if (fg || bold) {
      let style = '';
      if (fg)   style += `color:${fg};`;
      if (bold) style += 'font-weight:600;';
      out += `<span style="${style}">`;
      openSpans = 1;
    }
  }
  while (openSpans-- > 0) out += '</span>';
  return out;
}

// ── ANSI strip (for regex matching on raw streamed text) ──────────────────────
function stripAnsi(text) {
  // Strip all CSI escape sequences (colors, cursor movement, etc.)
  return text.replace(/\x1b\[[0-9;]*[A-Za-z]/g, '');
}

// ── Config helpers ────────────────────────────────────────────────────────────
function parseTotalIter(configContent) {
  const m = configContent.match(/^\s*total_iter\s*:\s*([\d_,]+)/m);
  return m ? parseInt(m[1].replace(/[_,]/g, ''), 10) : null;
}

function parsePrintFreq(configContent) {
  const m = configContent.match(/^\s*print_freq\s*:\s*(\d+)/m);
  return m ? parseInt(m[1], 10) : null;
}

function setIterProgressBar(pct) {
  const bar = $('iter-progress-bar');
  bar.classList.add('active');
  $('iter-progress-fill').style.width = pct + '%';
}

function clearIterProgressBar() {
  $('iter-progress-bar').classList.remove('active');
  $('iter-progress-fill').style.width = '0%';
}

// ── YAML syntax highlighter ───────────────────────────────────────────────────
function highlightYaml(code) {
  const e = s => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  const s = (cls, t) => `<span class="y-${cls}">${t}</span>`;

  return code.split('\n').map(raw => {
    if (!raw) return '';

    // Full-line comment
    if (/^\s*#/.test(raw)) return s('comment', e(raw));

    // YAML document / directive markers
    if (/^\s*(---|\.\.\.)\s*$/.test(raw)) return s('marker', e(raw));

    let out = '', rem = raw;

    // Leading whitespace
    const ind = raw.match(/^(\s*)/)[1];
    out += e(ind); rem = rem.slice(ind.length);

    // List-item dash "- "
    const dashM = rem.match(/^(-\s+)(.*)/);
    if (dashM) { out += s('dash', e(dashM[1])); rem = dashM[2]; }

    // Key: value
    // Require ": " or ":\s*$" so Windows paths like C:/foo don't look like keys
    const kvM = rem.match(/^([^:#"'\n][^:#"'\n]*)(\s*:\s+|\s*:\s*$)(.*)/);
    if (kvM) {
      out += s('key', e(kvM[1])) + s('colon', e(kvM[2]));
      rem = kvM[3];
    }

    // Value (remainder) + optional inline comment
    if (rem) {
      let val = rem, cmt = '';
      // Inline comment: find " #" not inside a quoted string
      if (!/^\s*['"]/.test(val)) {
        const ci = val.search(/\s+#/);
        if (ci > 0) { val = rem.slice(0, ci); cmt = s('comment', e(rem.slice(ci))); }
      }
      out += colorYamlValue(val, e, s) + cmt;
    }

    return out;
  }).join('\n');
}

function colorYamlValue(v, e, s) {
  const t = v.trim();
  if (!t) return e(v);
  // Leading whitespace before the value token
  const pre = e(v.slice(0, v.length - v.trimStart().length));
  if (/^'[^']*'$|^"[^"]*"$/.test(t))                         return pre + s('string', e(t));
  if (/^-?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$/.test(t))        return pre + s('number', e(t));
  if (/^(true|false|null|~|yes|no|on|off)$/i.test(t))         return pre + s('bool',   e(t));
  if (t.startsWith('!!'))                                      return pre + s('tag',    e(t));
  return e(v);
}

function updateHighlight() {
  const hl  = document.getElementById('yaml-highlight');
  const ed  = document.getElementById('config-editor');
  if (!hl || !ed) return;
  // Append a trailing newline so the pre is always at least as tall as the textarea
  hl.innerHTML = highlightYaml(ed.value) + '\n';
  // Sync position via transform — immune to scrollbar-space differences
  hl.style.transform = `translate(${-ed.scrollLeft}px, ${-ed.scrollTop}px)`;
}

// ── Validation popup ──────────────────────────────────────────────────────────
function showValidationPopup(pct, eta) {
  S.validating = true;
  clearIterProgressBar();
  const numVal = S.statsCache[S.trainingKey]?.numValImages;
  if (numVal) {
    const cur = Math.max(0, Math.round((pct / 100) * numVal));
    $('val-count').textContent = cur + ' / ' + numVal;
  } else {
    $('val-count').textContent = pct + '%';
  }
  $('val-eta').textContent   = eta ? 'ETA ' + eta : '';
  $('val-fill').style.width  = pct + '%';
  $('validation-popup').classList.add('visible');
}

function hideValidationPopup() {
  if (!S.validating) return;
  S.validating = false;
  $('validation-popup').classList.remove('visible');
}

// ── Training state detection ───────────────────────────────────────────────────
function detectState(text) {
  const plain = stripAnsi(text);
  const lo    = plain.toLowerCase();
  if (/saving model|saving.*state|save.*model/i.test(plain))                 return 'Saving Model';
  if (/saving\s+\d+\s+validation images/i.test(plain))                       return 'Validating';
  if (S.validating)                                                           return 'Validating';
  if (/\biter:\s*\d/.test(plain))                                            return 'Training';
  if (/number of (train|val) images|building.*dataset|dataloader/.test(lo))  return 'Building Dataset';
  if (/loading.*checkpoint|resuming|setting up/.test(lo))                    return 'Loading';
  if (/start training|begin training/.test(lo))                              return 'Training';
  return null;
}

function openConsole() {
  if (S._bh === 0) animateTo('bh', defaultBh());
}

function setTrainingStatus(state) {
  // Short-circuit when nothing changed — re-rendering innerHTML recreates
  // .spinner and restarts its CSS animation on every tqdm update.
  if (state === S.trainingStatus) return;
  // Freeze iter extrapolation and clear the progress bar when saving starts
  if (state === 'Saving Model' && S.trainingStatus !== 'Saving Model') {
    S.lastIterTime = 0;
    clearIterProgressBar();
  }
  S.trainingStatus = state;
  renderTrainingStatus();
}

function renderTrainingStatus() {
  const state = S.trainingStatus;
  const el = $('train-status');
  if (!state) { el.classList.add('hidden'); return; }
  el.classList.remove('hidden');

  // Show experiment name when: active + viewing another experiment,
  // or stopped/complete (always show for context,trainingKey is null by then)
  const viewingOther = S.trainingKey !== null && S.trainingKey !== S.selectedKey;
  const isTerminal   = state === 'complete' || state === 'stopped';
  const showName     = S.trainingName && (viewingOther || isTerminal);
  const subtext      = showName
    ? '<span class="status-subtext">(' + S.trainingName + ')</span>'
    : '';

  if (state === 'complete') {
    el.innerHTML =
      '<div class="status-text-group">' +
      '<span class="status-done">Training complete</span>' +
      subtext + '</div>';
  } else if (state === 'stopped') {
    el.innerHTML =
      '<div class="status-text-group">' +
      '<span class="status-stopped">traiNNer-redux process has stopped, ' +
      '<button class="console-link" onclick="openConsole()">check console</button>' +
      ' for details</span>' +
      subtext + '</div>';
  } else {
    // Active spinner states: Training / Loading / Building Dataset / Compiling / …
    el.innerHTML =
      '<div class="spinner"></div>' +
      '<div class="status-text-group"><span>' + state + '</span>' + subtext + '</div>';
  }
}

// ── Console output (CR-aware) ─────────────────────────────────────────────────
const MAX_CONSOLE_LINES = 2000;

function clearConsole() {
  S.consoleHTML       = '';
  S.consoleCurrentRaw = '';
  $('console-output').innerHTML = '';
}

function appendConsole(raw) {
  const out      = $('console-output');
  const atBottom = out.scrollHeight - out.scrollTop - out.clientHeight < 40;

  // Split on CR+LF, LF, or bare CR — keeping the delimiters
  const parts = raw.split(/(\r\n|\r|\n)/);
  for (const p of parts) {
    if (p === '\r\n' || p === '\n') {
      // Commit current line to the HTML buffer
      S.consoleHTML       += ansiToHtml(S.consoleCurrentRaw) + '\n';
      S.consoleCurrentRaw  = '';
      // Trim oldest lines to stay within MAX_CONSOLE_LINES
      let nl = 0;
      for (let i = 0; i < S.consoleHTML.length; i++) {
        if (S.consoleHTML[i] === '\n') nl++;
        if (nl > MAX_CONSOLE_LINES + 200) {
          // Drop the first 200 lines
          let drop = 0, j = 0;
          while (drop < 200 && j < S.consoleHTML.length) {
            if (S.consoleHTML[j++] === '\n') drop++;
          }
          S.consoleHTML = S.consoleHTML.slice(j);
          break;
        }
      }
    } else if (p === '\r') {
      S.consoleCurrentRaw = '';   // carriage return: overwrite current line
    } else if (p.length > 0) {
      S.consoleCurrentRaw += p;
    }
  }

  // Committed lines never change after being written; only re-render the live line
  out.innerHTML = S.consoleHTML + ansiToHtml(S.consoleCurrentRaw);
  if (atBottom) out.scrollTop = out.scrollHeight;
}

// ── WebSocket console ─────────────────────────────────────────────────────────
function connectWs(key) {
  if (S.ws) {
    S.ws.onclose = null;   // prevent old handler from firing when we close it
    S.ws.close();
    S.ws = null;
  }
  clearConsole();
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const ws    = new WebSocket(proto + '//' + location.host + '/ws/' + encodeURIComponent(key));

  ws.onmessage = (e) => {
    // Status control messages from server
    if (e.data.startsWith('\x00STATUS:')) {
      const status = e.data.slice(8).trim();
      S.trainingKey  = null;
      S.lastIterTime = 0;   // stop extrapolation
      hideValidationPopup();
      clearIterProgressBar();
      renderStats(S.selectedKey);
      updateTrainBtn();
      if (status === 'manual_stop') {
        setTrainingStatus(null);
      } else {
        setTrainingStatus(status);        // 'complete' or 'stopped'
      }
      return;
    }
    appendConsole(e.data);
    // Split on CR and LF so each tqdm \r-delimited update drives parseStats
    // independently — otherwise only the first (0%) match would be seen.
    const segments = e.data.split(/\r|\n/);
    for (const seg of segments) {
      if (!seg.trim()) continue;
      parseStats(seg);
      const st = detectState(seg);
      if (st) setTrainingStatus(st);
    }
  };
  // Only reset training state if this WS closes unexpectedly (not via our own close() call)
  ws.onclose = () => {
    if (S.trainingKey === key) {
      S.trainingKey  = null;
      S.lastIterTime = 0;   // stop extrapolation
      renderStats(S.selectedKey);
      updateTrainBtn();
      // Don't clear trainingStatus,let the last sentinel message (or its absence) stand
    }
  };
  S.ws = ws;
}

// ── Stats parser ──────────────────────────────────────────────────────────────
function parseStats(text) {
  // Strip ANSI codes first,Rich's ReprHighlighter wraps numbers in color
  // escape sequences that break plain regex matching.
  const plain = stripAnsi(text);
  const key   = S.trainingKey;
  if (!key) return;

  // Ensure a cache entry exists for this experiment
  if (!S.statsCache[key]) S.statsCache[key] = {};
  const c = S.statsCache[key];
  let changed = false;

  // Only match epoch from actual training log lines: [epoch:  1, iter:  ...
  // Avoids false positives from "Total epochs: 251" or "iter per epoch: 1,999"
  const epochM = plain.match(/\[epoch:\s*([\d,]+)/i);
  if (epochM) { c.epoch = parseInt(epochM[1].replace(/,/g, ''), 10); changed = true; }

  // Capture final iter from the save-on-exit line so lastIter is accurate when extrapolation stops
  // e.g. "Saving models and training states to experiments folder for epoch: 1, iter: 4932."
  const saveM = plain.match(/saving models.*\biter:\s*([\d,]+)/i);
  if (saveM) {
    c.iter = parseInt(saveM[1].replace(/,/g, ''), 10);
    S.lastIter = c.iter;
    S.lastIterTime = 0;   // freeze extrapolation at the checkpoint iter
    changed = true;
  }

  const iterM = plain.match(/\biter:\s*([\d,]+)/i);
  if (iterM) {
    c.iter = parseInt(iterM[1].replace(/,/g, ''), 10);
    S.lastIter     = c.iter;
    S.lastIterTime = Date.now();
    changed = true;
  }

  const itsM = plain.match(/(\d+\.?\d*)\s*it\/s/i);
  if (itsM) { c.its = parseFloat(itsM[1]); S.lastIts = c.its; changed = true; }

  const etaM = plain.match(/eta:\s*(\d{1,3}:\d{2}:\d{2})/i);
  if (etaM) { c.eta = etaM[1]; changed = true; }

  // lr:(1.000e-04) or lr:(1.000e-04, 2.000e-04),store first value as raw float
  const lrM = plain.match(/\blr:\(([^,)]+)/i);
  if (lrM) { c.lr_val = parseFloat(lrM[1]); changed = true; }

  const vramM = plain.match(/peak vram:\s*([\d.]+)\s*gb/i);
  if (vramM) { c.vram = parseFloat(vramM[1]); changed = true; }

  // Capture val dataset size from startup log or from the "Saving N validation images" line
  const valImgM = plain.match(/number of val images\/folders in [^:]+:\s*(\d+)/i);
  if (valImgM) c.numValImages = parseInt(valImgM[1], 10);

  // Detect validation start: "Saving 3 validation images to visualization folder."
  // This is the log line traiNNer emits right before the tqdm progress bar begins.
  const savingValM = plain.match(/saving\s+(\d+)\s+validation images/i);
  if (savingValM) {
    c.numValImages = parseInt(savingValM[1], 10);
    S.validating = true;
    showValidationPopup(0, null);
  }

  // Update progress while validating: tqdm outputs "Test N:  X% ..." lines (not "Validating:")
  if (S.validating && /\d+%/.test(plain) && !/\[epoch:/i.test(plain)) {
    // Use the last % match — a single chunk may contain multiple \r-delimited updates
    const pctAll = [...plain.matchAll(/(\d+)%/g)];
    // ETA format from tqdm: "< 0:00:03" (space after <)
    const etaM2  = plain.match(/<\s*([\d:]+)/);
    if (pctAll.length) {
      const pct = parseInt(pctAll[pctAll.length - 1][1], 10);
      showValidationPopup(pct, etaM2 ? etaM2[1] : null);
      // tqdm 100% means the validation loop has finished — let the user see
      // 100% briefly then hide.
      if (pct === 100) {
        // Immediately unblock the top bar (status + iter counter + progress bar)
        S.validating = false;
        setTrainingStatus('Training');
        S.lastIterTime = Date.now();
        // Keep the popup visible briefly so the user sees 100%
        // Call classList directly — S.validating is already false so
        // hideValidationPopup() would early-return without removing the class.
        setTimeout(() => $('validation-popup').classList.remove('visible'), 800);
        // Refresh viz check for the training experiment — new images may have been saved
        if (key && !(S.vizData[key] && S.vizData[key].folders.length)) {
          S.vizChecked.delete(key);
          checkVisualization(key);
        }
      }
    }
  }

  // Also hide on the [epoch:] line as a fallback (same brief delay so the
  // user always sees 100% before it disappears, regardless of training speed)
  if (S.validating && /\[epoch:/i.test(plain)) {
    S.validating = false;
    setTrainingStatus('Training');
    S.lastIterTime = Date.now();
    setTimeout(() => $('validation-popup').classList.remove('visible'), 800);
    if (key && !(S.vizData[key] && S.vizData[key].folders.length)) {
      S.vizChecked.delete(key);
      checkVisualization(key);
    }
  }

  // Update displays only when the user is currently viewing the training experiment
  if (changed && S.selectedKey === key) renderStats(key);

  // Debounce-persist stats to server (3 s after last update)
  if (changed) {
    clearTimeout(S.statsDebounce);
    S.statsDebounce = setTimeout(() => saveStatsToServer(key), 3000);
  }

  // Schedule a graph refresh ~2 s after each log line so TB data has time to flush to disk.
  // The debounce prevents hammering during rapid iteration bursts.
  if (changed) {
    clearTimeout(S._graphRefreshTimer);
    S._graphRefreshTimer = setTimeout(() => {
      if (S.trainingKey === S.selectedKey) loadGraphs(S.selectedKey);
    }, 2000);
  }
}

// ── Per-chart zoom / pin / expand state ──────────────────────────────────
const chartState = {};  // tag → { zoom:'all'|10|100|1000, offset:0, pinned:false, expanded:false }

function getChartState(tag) {
  if (!chartState[tag]) chartState[tag] = { zoom: 'all', offset: 0, pinned: false, expanded: false };
  return chartState[tag];
}

// Returns the slice of data to draw based on zoom+offset.
// offset=0 → newest N points; increasing offset scrolls toward older data.
function getVisibleData(data, st) {
  if (st.zoom === 'all' || data.length <= st.zoom) return data;
  const end   = data.length - st.offset;
  const start = Math.max(0, end - st.zoom);
  return data.slice(start, end);
}

// Sync all button states (active classes, disabled) for one card.
function syncCard(card, tag, data) {
  const st = getChartState(tag);
  const n  = data ? data.length : 0;

  const pinBtn = card.querySelector('.chart-pin-btn');
  if (pinBtn) {
    pinBtn.textContent = st.pinned ? '◉' : '◎';
    pinBtn.title       = st.pinned ? 'Unpin' : 'Pin to top';
    pinBtn.classList.toggle('active', st.pinned);
  }
  const expBtn = card.querySelector('.chart-exp-btn');
  if (expBtn) {
    expBtn.textContent = st.expanded ? '⊟' : '⊞';
    expBtn.title       = st.expanded ? 'Shrink' : 'Expand';
    expBtn.classList.toggle('active', st.expanded);
  }
  card.classList.toggle('pinned',   st.pinned);
  card.classList.toggle('expanded', st.expanded);

  for (const btn of card.querySelectorAll('.zoom-btn-z')) {
    const bz = btn.dataset.zoom;
    btn.classList.toggle('active', bz === 'all' ? st.zoom === 'all' : st.zoom === parseInt(bz, 10));
  }

  const navL = card.querySelector('.zoom-nav-l');
  const navR = card.querySelector('.zoom-nav-r');
  if (navL && navR) {
    const zoomed = st.zoom !== 'all';
    const nz     = zoomed ? st.zoom : 0;
    navL.disabled = !zoomed || n <= nz || st.offset >= n - nz;
    navR.disabled = !zoomed || st.offset <= 0;
    navL.style.opacity = zoomed ? '' : '0.18';
    navR.style.opacity = zoomed ? '' : '0.18';
  }

  // Keep custom input showing the live visible count (skip if user is actively typing)
  const custInp = card.querySelector('.zoom-custom-input');
  if (custInp && document.activeElement !== custInp) {
    const visCount = getVisibleData(data || [], st).length;
    custInp.value = visCount > 0 ? visCount : '';
  }
}

// Compute and apply card heights so charts fill the graphs-area.
// Normal cards share available height; expanded cards take 2× that height.
function resizeChartCards() {
  const area = document.getElementById('graphs-area');
  if (!area) return;
  const cards = [...area.querySelectorAll('.chart-card')];
  if (!cards.length) return;

  const gap    = 8;
  const pad    = 10;
  const innerH = area.clientHeight - pad * 2;
  const innerW = area.clientWidth  - pad * 2;
  const MIN_H  = 140;
  const MAX_H  = 560;

  // Match the CSS grid: auto-fill with minmax(300px, 1fr)
  const cols    = Math.max(1, Math.floor((innerW + gap) / (300 + gap)));
  const cardW   = Math.floor((innerW - gap * (cols - 1)) / cols);   // normal card width
  const expCardW = innerW;   // expanded cards span the full row

  const normCount = cards.filter(c => !c.classList.contains('expanded')).length;
  const expCount  = cards.filter(c =>  c.classList.contains('expanded')).length;
  const normRows  = Math.ceil(normCount / cols);
  const totalRows = normRows + expCount;   // each expanded card occupies a full row

  // Solve: normH×normRows + 2×normH×expCount + gap×(totalRows−1) = innerH
  const denom = normRows + 2 * expCount;
  let normH = denom > 0
    ? Math.floor((innerH - gap * Math.max(0, totalRows - 1)) / denom)
    : MIN_H;
  // Height must not exceed card width (no taller-than-wide cards)
  normH = Math.max(MIN_H, Math.min(MAX_H, normH, cardW));

  const expH = Math.min(MAX_H, normH * 2, expCardW);
  area.style.setProperty('--chart-h',     normH + 'px');
  area.style.setProperty('--chart-h-exp', expH  + 'px');
}

// ── Graph system ──────────────────────────────────────────────────────────────

async function loadGraphs(key) {
  if (!key) { showGraphsMessage('Select an experiment to view its graphs.'); return; }
  // Only show the loading spinner on initial load (no charts yet); skip it on
  // background polls so existing charts don't flash away every 3 seconds.
  const area = document.getElementById('graphs-area');
  if (!area.querySelector('.chart-card'))
    area.innerHTML = '<div class="graphs-empty"><div class="spinner"></div></div>';
  const res = await apiFetch('/api/experiments/' + encodeURIComponent(key) + '/graphs');
  if (!res) {
    if (key === S.selectedKey)
      showGraphsMessage('Server unreachable — graphs unavailable.');
    return;
  }
  const data = await res.json();
  if (key !== S.selectedKey) return;  // stale response, user switched away
  const scalars = data.scalars || {};
  if (!Object.keys(scalars).length) {
    const msg = data.error || 'No TensorBoard data yet — training will populate graphs here.';
    document.getElementById('graphs-area').innerHTML =
      '<div class="graphs-empty">' + msg + '</div>';
    return;
  }
  updateCharts(scalars);
}

function showGraphsMessage(msg) {
  document.getElementById('graphs-area').innerHTML =
    '<div class="graphs-empty">' + msg + '</div>';
}

function updateCharts(scalars) {
  const area = document.getElementById('graphs-area');
  const tags = Object.keys(scalars).sort();

  if (tags.length === 0) {
    showGraphsMessage('No TensorBoard data yet — training will populate graphs here.');
    return;
  }

  // Create / update cards
  const seen = new Set();
  for (const tag of tags) {
    seen.add(tag);
    let card = area.querySelector('.chart-card[data-tag="' + CSS.escape(tag) + '"]');
    if (!card) {
      card = createChartCard(tag);
      area.appendChild(card);
    }
    const cv = card.querySelector('.chart-canvas');
    cv._chartData = scalars[tag];
    syncCard(card, tag, cv._chartData);
  }

  // Remove cards whose tags disappeared
  for (const card of [...area.querySelectorAll('.chart-card')]) {
    if (!seen.has(card.dataset.tag)) card.remove();
  }

  // Clear placeholder if still present
  const empty = area.querySelector('.graphs-empty');
  if (empty) empty.remove();

  // Sort cards: pinned first (alphabetical), then unpinned (alphabetical)
  const allCards = [...area.querySelectorAll('.chart-card')];
  allCards.sort((a, b) => {
    const ap = getChartState(a.dataset.tag).pinned;
    const bp = getChartState(b.dataset.tag).pinned;
    if (ap !== bp) return ap ? -1 : 1;
    return a.dataset.tag.localeCompare(b.dataset.tag);
  });
  for (const c of allCards) area.appendChild(c);

  // Defer draws to next frame (after layout + CSS height vars are applied)
  // so canvases don't render against stale dimensions.
  redrawAllCharts();
}

function createChartCard(tag) {
  const card = document.createElement('div');
  card.className   = 'chart-card';
  card.dataset.tag = tag;

  // ── Header: title | pin | expand ──────────────────────────────────────
  const header = document.createElement('div');
  header.className = 'chart-header';

  const title = document.createElement('span');
  title.className = 'chart-title';
  const slashIdx = tag.lastIndexOf('/');
  if (slashIdx >= 0) {
    const prefix = document.createTextNode(tag.slice(0, slashIdx + 1));
    const leaf   = document.createElement('span');
    leaf.className   = 'chart-title-leaf';
    leaf.textContent = tag.slice(slashIdx + 1);
    title.appendChild(prefix);
    title.appendChild(leaf);
  } else {
    const leaf = document.createElement('span');
    leaf.className   = 'chart-title-leaf';
    leaf.textContent = tag;
    title.appendChild(leaf);
  }

  const pinBtn = document.createElement('button');
  pinBtn.className = 'chart-icon-btn chart-pin-btn';

  const expBtn = document.createElement('button');
  expBtn.className = 'chart-icon-btn chart-exp-btn';

  header.appendChild(title);
  header.appendChild(pinBtn);
  header.appendChild(expBtn);

  // ── Canvas ─────────────────────────────────────────────────────────────
  const cv = document.createElement('canvas');
  cv.className  = 'chart-canvas';
  cv._chartData = [];
  cv._hoverIdx  = null;
  cv._tag       = tag;

  // ── Tooltip ────────────────────────────────────────────────────────────
  const tip = document.createElement('div');
  tip.className = 'chart-tooltip hidden';

  // ── Zoom / nav bar ─────────────────────────────────────────────────────
  const zoomBar = document.createElement('div');
  zoomBar.className = 'chart-zoom-bar';

  const navL = document.createElement('button');
  navL.className   = 'zoom-btn zoom-nav zoom-nav-l';
  navL.title       = 'Scroll to older data';
  navL.textContent = '◀';

  const btnGroup = document.createElement('div');
  btnGroup.className = 'zoom-btn-group';

  const ZOOM_LEVELS = [['all','All'],['1000','1000'],['500','500'],['250','250'],['100','100'],['50','50']];
  const zBtns = ZOOM_LEVELS.map(([z, label]) => {
    const b = document.createElement('button');
    b.className    = 'zoom-btn zoom-btn-z';
    b.dataset.zoom = z;
    b.textContent  = label;
    return b;
  });
  for (const b of zBtns) btnGroup.appendChild(b);

  // ── Custom count input + ▲▼ spin buttons ──────────────────────────────
  const customWrap = document.createElement('div');
  customWrap.className = 'zoom-custom';

  const custInp = document.createElement('input');
  custInp.className   = 'zoom-custom-input';
  custInp.type        = 'text';
  custInp.inputMode   = 'numeric';
  custInp.title       = 'Custom datapoint count — press Enter to apply';

  const spinBtns = document.createElement('div');
  spinBtns.className = 'zoom-spin-btns';

  const spinUp = document.createElement('button');
  spinUp.className   = 'zoom-btn zoom-spin-up';
  spinUp.textContent = '▲';
  spinUp.title       = 'Show more datapoints (×1.1)';

  const spinDn = document.createElement('button');
  spinDn.className   = 'zoom-btn zoom-spin-dn';
  spinDn.textContent = '▼';
  spinDn.title       = 'Show fewer datapoints (÷1.1)';

  spinBtns.appendChild(spinUp);
  spinBtns.appendChild(spinDn);
  customWrap.appendChild(custInp);
  customWrap.appendChild(spinBtns);

  const navR = document.createElement('button');
  navR.className   = 'zoom-btn zoom-nav zoom-nav-r';
  navR.title       = 'Scroll to newer data';
  navR.textContent = '▶';

  zoomBar.appendChild(navL);
  zoomBar.appendChild(btnGroup);
  zoomBar.appendChild(customWrap);
  zoomBar.appendChild(navR);

  card.appendChild(header);
  card.appendChild(cv);
  card.appendChild(tip);
  card.appendChild(zoomBar);

  // ── Hover on canvas ────────────────────────────────────────────────────
  cv.addEventListener('mousemove', e => {
    const data = getVisibleData(cv._chartData || [], getChartState(tag));
    if (!data.length) return;
    const rect = cv.getBoundingClientRect();
    const mx   = e.clientX - rect.left;
    const P    = CHART_PAD;
    const cW   = cv.offsetWidth - P.l - P.r;
    const x0   = data[0].step, x1 = data[data.length - 1].step;
    if (x1 === x0) { cv._hoverIdx = 0; }
    else {
      const targetStep = x0 + ((mx - P.l) / cW) * (x1 - x0);
      let best = 0, bestDist = Infinity;
      for (let i = 0; i < data.length; i++) {
        const d = Math.abs(data[i].step - targetStep);
        if (d < bestDist) { bestDist = d; best = i; }
      }
      cv._hoverIdx = best;
    }
    drawChart(cv);
    const pt = data[cv._hoverIdx];
    tip.textContent = 'step ' + pt.step.toLocaleString() + ':  ' + fmtVal(pt.value);
    tip.classList.remove('hidden');
  });

  cv.addEventListener('mouseleave', () => {
    cv._hoverIdx = null;
    drawChart(cv);
    tip.classList.add('hidden');
  });

  // ── Pin button ─────────────────────────────────────────────────────────
  pinBtn.addEventListener('click', () => {
    getChartState(tag).pinned = !getChartState(tag).pinned;
    syncCard(card, tag, cv._chartData);
    // Re-sort all cards: pinned first, then alphabetical
    const area     = document.getElementById('graphs-area');
    const allCards = [...area.querySelectorAll('.chart-card')];
    allCards.sort((a, b) => {
      const ap = getChartState(a.dataset.tag).pinned;
      const bp = getChartState(b.dataset.tag).pinned;
      if (ap !== bp) return ap ? -1 : 1;
      return a.dataset.tag.localeCompare(b.dataset.tag);
    });
    for (const c of allCards) area.appendChild(c);
  });

  // ── Expand button ──────────────────────────────────────────────────────
  expBtn.addEventListener('click', () => {
    getChartState(tag).expanded = !getChartState(tag).expanded;
    syncCard(card, tag, cv._chartData);
    resizeChartCards();
    requestAnimationFrame(() => drawChart(cv));
  });

  // ── Zoom buttons ───────────────────────────────────────────────────────
  for (const b of zBtns) {
    b.addEventListener('click', () => {
      const st  = getChartState(tag);
      const raw = b.dataset.zoom;
      st.zoom   = raw === 'all' ? 'all' : parseInt(raw, 10);
      st.offset = 0;   // reset to newest end
      syncCard(card, tag, cv._chartData);
      drawChart(cv);
    });
  }

  // ── Custom input ──────────────────────────────────────────────────────
  function applyCustomZoom(raw) {
    const n = parseInt(raw, 10);
    if (isNaN(n) || n < 1) return;
    const st   = getChartState(tag);
    const data = cv._chartData || [];
    st.zoom    = n;
    st.offset  = Math.min(st.offset, Math.max(0, data.length - n));
    syncCard(card, tag, data);
    drawChart(cv);
  }

  custInp.addEventListener('keydown', e => {
    if (e.key === 'Enter')  { applyCustomZoom(custInp.value); custInp.blur(); }
    if (e.key === 'Escape') { custInp.blur(); }
  });
  custInp.addEventListener('blur', () => applyCustomZoom(custInp.value));

  // ── Spin buttons (×/÷ 1.1) ────────────────────────────────────────────
  spinUp.addEventListener('click', () => {
    const st   = getChartState(tag);
    const data = cv._chartData || [];
    const cur  = st.zoom === 'all' ? data.length : st.zoom;
    const next = Math.round(cur * 1.1);
    if (next >= data.length) {
      st.zoom = 'all';
    } else {
      st.zoom = Math.max(1, next);
    }
    st.offset = 0;
    syncCard(card, tag, data);
    drawChart(cv);
  });

  spinDn.addEventListener('click', () => {
    const st   = getChartState(tag);
    const data = cv._chartData || [];
    const cur  = st.zoom === 'all' ? data.length : st.zoom;
    const next = Math.max(1, Math.round(cur / 1.1));
    st.zoom    = next;
    st.offset  = Math.min(st.offset, Math.max(0, data.length - next));
    syncCard(card, tag, data);
    drawChart(cv);
  });

  // ── Nav buttons (scroll by 1/10th of zoom window) ─────────────────────
  navL.addEventListener('click', () => {
    const st   = getChartState(tag);
    const data = cv._chartData || [];
    if (st.zoom === 'all') return;
    const step   = Math.max(1, Math.floor(st.zoom / 10));
    const maxOff = Math.max(0, data.length - st.zoom);
    st.offset    = Math.min(maxOff, st.offset + step);
    syncCard(card, tag, data);
    drawChart(cv);
  });

  navR.addEventListener('click', () => {
    const st = getChartState(tag);
    if (st.zoom === 'all') return;
    const step = Math.max(1, Math.floor(st.zoom / 10));
    st.offset  = Math.max(0, st.offset - step);
    syncCard(card, tag, cv._chartData);
    drawChart(cv);
  });

  syncCard(card, tag, []);
  return card;
}

const CHART_PAD = { t: 6, r: 10, b: 22, l: 56 };

function drawChart(cv) {
  const allData = cv._chartData || [];
  const data    = cv._tag ? getVisibleData(allData, getChartState(cv._tag)) : allData;
  const dpr  = window.devicePixelRatio || 1;
  const W    = cv.offsetWidth, H = cv.offsetHeight;
  if (!W || !H) return;
  cv.width  = W * dpr;
  cv.height = H * dpr;
  const ctx = cv.getContext('2d');
  ctx.scale(dpr, dpr);

  const P  = CHART_PAD;
  const cW = W - P.l - P.r;
  const cH = H - P.t - P.b;

  ctx.fillStyle = SURF;
  ctx.fillRect(0, 0, W, H);

  if (!data.length) {
    ctx.fillStyle = MUTED; ctx.font = '11px system-ui';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('No data', W / 2, H / 2);
    return;
  }

  const x0 = data[0].step, x1 = data[data.length - 1].step;
  let y0 = Infinity, y1 = -Infinity;
  for (const d of data) { if (d.value < y0) y0 = d.value; if (d.value > y1) y1 = d.value; }
  if (y0 === y1) { const pad = Math.abs(y0) * 0.05 || 0.01; y0 -= pad; y1 += pad; }

  const sx = v => P.l + (x1 === x0 ? cW / 2 : (v - x0) / (x1 - x0) * cW);
  const sy = v => P.t + (1 - (v - y0) / (y1 - y0)) * cH;

  // Y-axis grid lines + labels
  const yTicks = niceTicks(y0, y1, 4);
  ctx.strokeStyle = BORDER; ctx.lineWidth = 1;
  ctx.font = '9px system-ui'; ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  ctx.fillStyle = MUTED;
  for (const v of yTicks) {
    const y = sy(v);
    ctx.beginPath(); ctx.moveTo(P.l, y); ctx.lineTo(P.l + cW, y); ctx.stroke();
    ctx.fillText(fmtVal(v), P.l - 4, y);
  }

  // X-axis labels
  ctx.textAlign = 'center'; ctx.textBaseline = 'top'; ctx.fillStyle = MUTED;
  if (x1 > x0) {
    for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(cW / 70)))) {
      ctx.fillText(fmtStep(v), sx(v), P.t + cH + 4);
    }
  }

  // Data line
  ctx.strokeStyle = ACC; ctx.lineWidth = 1.5;
  ctx.beginPath();
  for (let i = 0; i < data.length; i++) {
    const x = sx(data[i].step), y = sy(data[i].value);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  }
  ctx.stroke();

  // Hover: crosshair + dot
  const hi = cv._hoverIdx;
  if (hi !== null) {
    const hx = sx(data[hi].step), hy = sy(data[hi].value);
    ctx.strokeStyle = 'rgba(249,250,251,0.2)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx, P.t); ctx.lineTo(hx, P.t + cH); ctx.stroke();
    ctx.fillStyle = ACC;
    ctx.beginPath(); ctx.arc(hx, hy, 3.5, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = TEXT; ctx.lineWidth = 1.5; ctx.stroke();
  }
}

let _chartRaf = null;
function redrawAllCharts() {
  // Set CSS height variables immediately so the browser starts computing layout,
  // then defer the actual pixel reads + draws to the next frame when layout has settled.
  resizeChartCards();
  if (_chartRaf) cancelAnimationFrame(_chartRaf);
  _chartRaf = requestAnimationFrame(() => {
    resizeChartCards();   // re-measure now that layout is committed
    for (const cv of document.querySelectorAll('.chart-canvas')) drawChart(cv);
    _chartRaf = null;
  });
}

// Tick generation helpers
function niceNum(x, round) {
  const e = Math.floor(Math.log10(x));
  const f = x / 10 ** e;
  let n;
  if (round) n = f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10;
  else        n = f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10;
  return n * 10 ** e;
}

function niceTicks(lo, hi, n) {
  if (lo === hi || !isFinite(lo) || !isFinite(hi)) return [];
  const step  = niceNum(niceNum(hi - lo, false) / n, true);
  const start = Math.ceil(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 1e-6; v += step)
    ticks.push(parseFloat(v.toPrecision(10)));
  return ticks;
}

function fmtVal(v) {
  if (v === 0) return '0';
  const a = Math.abs(v);
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  if (a < 0.1)  return v.toFixed(4);
  if (a < 10)   return v.toFixed(3);
  if (a < 1000) return v.toFixed(1);
  return v.toFixed(0);
}

function fmtStep(v) {
  if (v >= 1e6) return (v / 1e6).toFixed(1).replace(/\.0$/, '') + 'M';
  if (v >= 1e3) return (v / 1e3).toFixed(0) + 'k';
  return String(Math.round(v));
}

// ── Visualization viewer ──────────────────────────────────────────────────────

const V = {
  key:       null,
  folders:   [],     // [{name, images:[]}]
  folderIdx: 0,
  imgIdx:    0,
  mode:      'sr',   // 'sr' | 'lq' | 'gt'
  srW: 0, srH: 0,    // natural pixel size of the current SR image
  pan: { x: 0, y: 0 },
  drag: null,        // {sx,sy,px,py} while mouse is held
};

async function checkVisualization(key) {
  if (!key) { updateVizButton(); return; }
  if (!S.vizChecked.has(key)) {
    S.vizChecked.add(key);
    const res = await apiFetch('/api/experiments/' + encodeURIComponent(key) + '/visualization');
    if (res && res.ok) {
      const d = await res.json();
      S.vizData[key] = d;
    }
  }
  updateVizButton();
}

function updateVizButton() {
  const key = S.selectedKey;
  const has = !!(key && S.vizData[key] && S.vizData[key].folders.length);
  document.getElementById('viz-eye-wrap').classList.toggle('visible', has);
}

function openVizViewer() {
  const key = S.selectedKey;
  if (!key || !S.vizData[key]) return;
  V.key       = key;
  V.folders   = S.vizData[key].folders;
  V.folderIdx = 0;
  V.mode      = 'sr';
  // Default: last image (latest) of first folder
  V.imgIdx = Math.max(0, V.folders[0].images.length - 1);
  document.getElementById('viz-overlay').classList.remove('hidden');
  renderVizFolders();
  loadVizImage();
  initVizPan();
}

function closeVizViewer() {
  document.getElementById('viz-overlay').classList.add('hidden');
}

function renderVizFolders() {
  const list = document.getElementById('viz-folder-list');
  list.innerHTML = '';
  V.folders.forEach((f, i) => {
    const item = document.createElement('div');
    item.className = 'viz-folder-item' + (i === V.folderIdx ? ' active' : '');
    item.title = f.name + ' — ' + f.images.length + ' image' + (f.images.length !== 1 ? 's' : '');

    const thumb = document.createElement('img');
    thumb.className = 'viz-folder-thumb';
    thumb.src = vizImageUrl(f.name, f.images[f.images.length - 1]);
    thumb.alt = '';

    const nameEl = document.createElement('span');
    nameEl.className = 'viz-folder-name';
    nameEl.textContent = f.name;

    item.appendChild(thumb);
    item.appendChild(nameEl);
    item.addEventListener('click', () => {
      V.folderIdx = i;
      V.imgIdx    = Math.max(0, V.folders[i].images.length - 1);
      renderVizFolders();
      loadVizImage();
    });
    list.appendChild(item);
  });
}

function vizImageUrl(folder, filename) {
  return '/api/experiments/' + encodeURIComponent(V.key)
    + '/visualization/' + encodeURIComponent(folder)
    + '/' + encodeURIComponent(filename);
}

function loadVizImage() {
  const folder  = V.folders[V.folderIdx];
  if (!folder) return;
  const srFile  = folder.images[V.imgIdx];
  if (!srFile) return;

  const img     = document.getElementById('viz-img');
  const fn      = document.getElementById('viz-filename');
  fn.textContent = srFile;

  // Reset pan each time a new image is loaded
  V.pan = { x: 0, y: 0 };

  let url;
  if (V.mode === 'lq') {
    const lqFile = folder.lq_map?.[srFile];
    if (lqFile) {
      url = vizImageUrl(folder.name, lqFile);
    } else {
      // No LR file known for this image — stay on SR
      V.mode = 'sr';
      updateVizModeButtons();
      url = vizImageUrl(folder.name, srFile);
    }
  } else if (V.mode === 'gt') {
    url = '/api/experiments/' + encodeURIComponent(V.key)
        + '/gt-image/' + encodeURIComponent(srFile);
  } else {
    url = vizImageUrl(folder.name, srFile);
  }

  img.onload = () => {
    if (V.mode === 'sr') { V.srW = img.naturalWidth; V.srH = img.naturalHeight; }
    // For LR: scale up to match SR dimensions
    if (V.mode === 'lq' && V.srW && V.srH) {
      img.style.width  = V.srW + 'px';
      img.style.height = V.srH + 'px';
    } else {
      img.style.width  = '';
      img.style.height = '';
    }
    applyVizPan();
  };
  img.onerror = () => {
    // GT not found — fall back to SR and show a note in the filename area
    if (V.mode === 'gt') {
      document.getElementById('viz-filename').textContent = srFile + '  (GT not found)';
      V.mode = 'sr';
      updateVizModeButtons();
      img.src = vizImageUrl(folder.name, srFile);
    }
  };
  img.src = url;

  updateVizModeButtons();
}

function updateVizModeButtons() {
  const folder = V.folders[V.folderIdx];
  const srFile = folder?.images[V.imgIdx];
  document.getElementById('viz-lq-btn').disabled = !folder?.lq_map?.[srFile];
  ['sr','lq','gt'].forEach(m => {
    document.getElementById('viz-' + m + '-btn').classList.toggle('active', V.mode === m);
  });
  // Update nav button enabled state
  const images = V.folders[V.folderIdx]?.images ?? [];
  document.querySelectorAll('.viz-nav-btn').forEach(btn => {
    const step = parseInt(btn.dataset.step, 10);
    btn.disabled = (step < 0 && V.imgIdx === 0) || (step > 0 && V.imgIdx === images.length - 1);
  });
}

function applyVizPan() {
  const img = document.getElementById('viz-img');
  const vp  = document.getElementById('viz-viewport');
  const vpW = vp.clientWidth;
  const vpH = vp.clientHeight;
  const iW  = img.offsetWidth  || img.naturalWidth;
  const iH  = img.offsetHeight || img.naturalHeight;
  // Center when image fits the axis; top/left-anchor when larger (panning)
  const baseX = iW <= vpW ? (vpW - iW) / 2 : 0;
  const baseY = iH <= vpH ? (vpH - iH) / 2 : 0;
  img.style.left = (baseX + V.pan.x) + 'px';
  img.style.top  = (baseY + V.pan.y) + 'px';
}

function initVizPan() {
  const vp = document.getElementById('viz-viewport');

  vp.onmousedown = (e) => {
    if (e.button !== 0) return;
    e.preventDefault();
    V.drag = { sx: e.clientX, sy: e.clientY, px: V.pan.x, py: V.pan.y };
    vp.classList.add('dragging');
  };

  vp.onmousemove = (e) => {
    if (!V.drag) return;
    const img = document.getElementById('viz-img');
    const vpW = vp.clientWidth;
    const vpH = vp.clientHeight;
    const iW  = img.offsetWidth  || ((V.mode === 'lq' && V.srW) ? V.srW : img.naturalWidth);
    const iH  = img.offsetHeight || ((V.mode === 'lq' && V.srH) ? V.srH : img.naturalHeight);
    const dx  = e.clientX - V.drag.sx;
    const dy  = e.clientY - V.drag.sy;
    V.pan.x   = Math.min(0, Math.max(V.drag.px + dx, vpW - iW));
    V.pan.y   = Math.min(0, Math.max(V.drag.py + dy, vpH - iH));
    applyVizPan();
  };

  const stopDrag = () => { V.drag = null; vp.classList.remove('dragging'); };
  vp.onmouseup   = stopDrag;
  vp.onmouseleave = stopDrag;
}

// ── Init ──────────────────────────────────────────────────────────────────────
function init() {
  // Load persisted sizes (or compute defaults from window dimensions)
  loadLayout();
  // Run through setters so push-through logic validates against actual window size
  setLw(S._lw); setRw(S._rw); setBh(S._bh);

  relayout();

  window.addEventListener('resize', relayout);

  // Tabs,open value is the current stored size (or default if closed)
  setupTab('tab-left',   'lw', S._lw || defaultLw());
  setupTab('tab-right',  'rw', S._rw || defaultRw());
  setupTab('tab-bottom', 'bh', S._bh || defaultBh());

  // Handles
  setupHandle('handle-left',   'lw');
  setupHandle('handle-right',  'rw');
  setupHandle('handle-bottom', 'bh');

  // New experiment button
  $('new-experiment-btn').addEventListener('click', showNewExperimentModal);

  // Sort dropdown
  const sortSel = document.getElementById('exp-sort-select');
  sortSel.value = getSortOrder();
  sortSel.addEventListener('change', () => {
    localStorage.setItem(SORT_KEY, sortSel.value);
    renderExperiments();
  });

  // Context menu actions
  $('ctx-new').addEventListener('click', (e) => {
    e.stopPropagation();
    $('ctx-menu').classList.add('hidden');
    showNewExperimentModal();
  });
  $('ctx-rename').addEventListener('click', (e) => {
    e.stopPropagation();
    $('ctx-menu').classList.add('hidden');
    if (!ctxKey) return;
    showRenameModal(ctxKey);
  });
  $('ctx-duplicate').addEventListener('click', async (e) => {
    e.stopPropagation();
    $('ctx-menu').classList.add('hidden');
    if (!ctxKey) return;
    await fetch('/api/experiments/' + encodeURIComponent(ctxKey) + '/duplicate', { method: 'POST' });
    await loadExperiments();
  });
  $('ctx-delete').addEventListener('click', async (e) => {
    e.stopPropagation();
    $('ctx-menu').classList.add('hidden');
    if (!ctxKey) return;
    if (!confirm('Delete experiment "' + ctxKey + '"?')) return;
    await fetch('/api/experiments/' + encodeURIComponent(ctxKey), { method: 'DELETE' });
    if (S.selectedKey === ctxKey) {
      S.selectedKey = null;
      $('config-editor').value = '';
      updateTrainBtn();
    }
    await loadExperiments();
  });

  // New experiment modal
  $('modal-overlay').addEventListener('click', (e) => { if (e.target === $('modal-overlay')) closeModal(); });
  $('modal-cancel').addEventListener('click', closeModal);
  $('modal-ok').addEventListener('click', createExperiment);
  $('modal-arch').addEventListener('change', (e) => loadTemplates(e.target.value));
  $('modal-template').addEventListener('change', (e) => { $('modal-name').value = e.target.value; });

  // Rename modal
  $('rename-overlay').addEventListener('click', (e) => { if (e.target === $('rename-overlay')) closeRenameModal(); });
  $('rename-cancel').addEventListener('click', closeRenameModal);
  $('rename-ok').addEventListener('click', submitRename);

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { closeModal(); closeRenameModal(); closeVizViewer(); }
    if (!$('viz-overlay').classList.contains('hidden')) {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') {
        e.preventDefault();
        const images = V.folders[V.folderIdx]?.images ?? [];
        V.imgIdx = Math.max(0, Math.min(images.length - 1, V.imgIdx + (e.key === 'ArrowLeft' ? -10 : 10)));
        loadVizImage();
      } else if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
        e.preventDefault();
        V.folderIdx = Math.max(0, Math.min(V.folders.length - 1, V.folderIdx + (e.key === 'ArrowUp' ? -1 : 1)));
        V.imgIdx = Math.max(0, V.folders[V.folderIdx].images.length - 1);
        renderVizFolders();
        loadVizImage();
      }
    }
    if ((e.ctrlKey || e.metaKey) && e.key === 's') { e.preventDefault(); saveConfig(); }
    if (e.key === 'Enter') {
      if (!$('modal-overlay').classList.contains('hidden')) {
        e.preventDefault(); createExperiment();
      } else if (!$('rename-overlay').classList.contains('hidden')) {
        e.preventDefault(); submitRename();
      } else if ((e.ctrlKey || e.metaKey) && e.altKey) {
        // Ctrl+Alt+Enter — stop training
        e.preventDefault();
        if (S.trainingKey && S.trainingKey === S.selectedKey) toggleTraining();
      } else if ((e.ctrlKey || e.metaKey)) {
        // Ctrl+Enter — start training
        e.preventDefault();
        if (!S.trainingKey && S.selectedKey) toggleTraining();
      }
    }
  });

  // Config editor,YAML highlighting + dirty detection
  const ed = $('config-editor');
  ed.addEventListener('input', () => {
    updateHighlight();
    setConfigDirty(S.originalConfig !== null && ed.value !== S.originalConfig);
  });
  ed.addEventListener('scroll', () => {
    const hl = document.getElementById('yaml-highlight');
    hl.style.transform = `translate(${-ed.scrollLeft}px, ${-ed.scrollTop}px)`;
  });

  // Config save bar
  $('save-config-btn').addEventListener('click', saveConfig);
  $('discard-config-btn').addEventListener('click', discardConfig);
  $('train-btn').addEventListener('click', toggleTraining);

  // Visualization viewer
  $('viz-eye-btn').addEventListener('click', openVizViewer);
  $('viz-close-btn').addEventListener('click', closeVizViewer);
  $('viz-overlay').addEventListener('click', (e) => { if (e.target === $('viz-overlay')) closeVizViewer(); });
  ['sr','lq','gt'].forEach(m => {
    document.getElementById('viz-' + m + '-btn').addEventListener('click', () => {
      V.mode = m;
      loadVizImage();
    });
  });
  document.querySelectorAll('.viz-nav-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const images = V.folders[V.folderIdx]?.images ?? [];
      V.imgIdx = Math.max(0, Math.min(images.length - 1, V.imgIdx + parseInt(btn.dataset.step, 10)));
      loadVizImage();
    });
  });

  // Extrapolate iteration count between log lines using last known it/s
  // Paused during validation and after stop is pressed
  setInterval(() => {
    if (S.trainingKey && S.trainingKey === S.selectedKey && S.lastIts > 0 && S.lastIterTime > 0 && !S.validating && S.trainingStatus !== 'Saving Model') {
      const elapsed   = (Date.now() - S.lastIterTime) / 1000;
      const estimated = S.lastIter + Math.floor(elapsed * S.lastIts);
      $('stat-iter').textContent = estimated.toLocaleString();
      // Drive the print_freq progress bar
      const printFreq = S.statsCache[S.selectedKey]?.printFreq;
      if (printFreq) setIterProgressBar((estimated % printFreq) / printFreq * 100);
    }
  }, 250);

  // Background graph poll,3 s is much better than TensorBoard's 30 s
  setInterval(() => {
    if (S.selectedKey) loadGraphs(S.selectedKey);
  }, 3000);

  // Redraw charts when the window / panel is resized
  new ResizeObserver(redrawAllCharts).observe(document.getElementById('graphs-area'));

  updateTrainBtn();
  loadExperiments();
}

document.addEventListener('DOMContentLoaded', init);
