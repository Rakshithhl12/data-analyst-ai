/**
 * DataAnalyst AI — Shared JavaScript Utilities
 * Works across all pages.
 */

// ── Toast ─────────────────────────────────────────────────────────────────────
function toast(msg, type = 'success', duration = 4200) {
  let container = document.getElementById('toast-container');
  if (!container) {
    container = document.createElement('div');
    container.id = 'toast-container';
    container.className = 'toast-container';
    document.body.appendChild(container);
  }
  const el = document.createElement('div');
  el.className = `toast ${type}`;
  const icons = { success: '✅', error: '❌', warning: '⚠️', info: 'ℹ️' };
  el.innerHTML = `<span>${icons[type] || ''}</span><span>${msg}</span>`;
  container.appendChild(el);
  setTimeout(() => {
    el.style.opacity = '0';
    el.style.transform = 'translateX(30px)';
    el.style.transition = 'all 0.3s';
    setTimeout(() => el.remove(), 300);
  }, duration);
}

// ── API helpers ───────────────────────────────────────────────────────────────
const API = {
  async post(url, data) {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `Request failed (${res.status})`);
    return json;
  },
  async get(url) {
    const res = await fetch(url);
    const json = await res.json();
    if (!res.ok) throw new Error(json.error || `Request failed (${res.status})`);
    return json;
  },
  async del(url) {
    const res = await fetch(url, { method: 'DELETE' });
    return res.ok;
  }
};

// ── Session helpers ───────────────────────────────────────────────────────────
const Session = {
  get()    { return localStorage.getItem('ada_session_id'); },
  set(id)  { localStorage.setItem('ada_session_id', id); },
  clear()  { localStorage.removeItem('ada_session_id'); }
};

// ── Auth ──────────────────────────────────────────────────────────────────────
async function logout() {
  try { await API.post('/api/logout', {}); } catch (_) {}
  Session.clear();
  window.location.href = '/login';
}

async function loadUserName(elId = 'nav-name') {
  try {
    const d = await API.get('/api/me');
    if (d.authenticated) {
      const el = document.getElementById(elId);
      if (el) el.textContent = d.name;
    }
  } catch (_) {}
}

// ── Mobile nav toggle ─────────────────────────────────────────────────────────
function initMobileNav() {
  const toggle = document.getElementById('nav-toggle');
  const menu   = document.getElementById('mobile-menu');
  if (!toggle || !menu) return;
  toggle.addEventListener('click', () => {
    menu.classList.toggle('open');
    toggle.innerHTML = menu.classList.contains('open')
      ? '<i class="fas fa-times"></i>'
      : '<i class="fas fa-bars"></i>';
  });
  // Close on outside click
  document.addEventListener('click', e => {
    if (!toggle.contains(e.target) && !menu.contains(e.target)) {
      menu.classList.remove('open');
      toggle.innerHTML = '<i class="fas fa-bars"></i>';
    }
  });
}

// ── Number formatters ─────────────────────────────────────────────────────────
function fmt(v, decimals = 2) {
  if (v == null || v === '') return '—';
  const n = parseFloat(v);
  if (isNaN(n)) return String(v);
  return n.toLocaleString(undefined, {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals
  });
}
function fmtInt(v) {
  if (v == null) return '—';
  return parseInt(v).toLocaleString();
}

// ── Correlation strength label ────────────────────────────────────────────────
function corrStrength(r) {
  const a = Math.abs(r);
  if (a >= 0.9) return 'Very Strong';
  if (a >= 0.7) return 'Strong';
  if (a >= 0.5) return 'Moderate';
  return 'Weak';
}

// ── Pipeline animator (index page) ───────────────────────────────────────────
function litPipeline(upTo) {
  for (let i = 0; i < 8; i++)
    document.getElementById('pb-' + i)?.classList.toggle('lit', i <= upTo);
  for (let i = 0; i < 7; i++)
    document.getElementById('pa-' + i)?.classList.toggle('lit', i < upTo);
}

// ── Progress bar helper ───────────────────────────────────────────────────────
function animateProgress(barId, from, to, step = 2, intervalMs = 25) {
  let v = from;
  const t = setInterval(() => {
    v = Math.min(v + step, to);
    const el = document.getElementById(barId);
    if (el) el.style.width = v + '%';
    if (v >= to) clearInterval(t);
  }, intervalMs);
  return t;
}

// ── DOM helpers ───────────────────────────────────────────────────────────────
function show(id) { const el = document.getElementById(id); if (el) el.style.display = ''; }
function hide(id) { const el = document.getElementById(id); if (el) el.style.display = 'none'; }

// ── Table builder ─────────────────────────────────────────────────────────────
function buildTable(headers, rows) {
  if (!rows || !rows.length) return '<p class="text-muted">No data.</p>';
  const ths = headers.map(h => `<th>${h}</th>`).join('');
  const trs = rows.map(r => '<tr>' + r.map(c => `<td>${c ?? '—'}</td>`).join('') + '</tr>').join('');
  return `<div class="table-wrap"><table class="ada-table"><thead><tr>${ths}</tr></thead><tbody>${trs}</tbody></table></div>`;
}

// ── Quality gauge renderer ────────────────────────────────────────────────────
function renderQualityGauge(score, arcId, scoreId, labelId, descId, descText = '') {
  const CIRC = 226;
  const color = score >= 80 ? '#22c55e' : score >= 60 ? '#f59e0b' : '#ef4444';
  const label = score >= 90 ? 'Excellent' : score >= 80 ? 'Good' : score >= 60 ? 'Fair' : 'Poor';
  const dash  = (CIRC * score / 100).toFixed(2);

  const arc   = document.getElementById(arcId);
  const sc    = document.getElementById(scoreId);
  const lb    = document.getElementById(labelId);
  const dc    = document.getElementById(descId);

  if (arc) { arc.setAttribute('stroke', color); arc.setAttribute('stroke-dasharray', `${dash} ${CIRC}`); }
  if (sc)  { sc.textContent = score; sc.style.color = color; }
  if (lb)  { lb.textContent = label + ' Quality'; lb.style.color = color; }
  if (dc)  { dc.textContent = descText; }
}

// Init on DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
  initMobileNav();
  loadUserName('nav-name');
});

window.toast       = toast;
window.API         = API;
window.Session     = Session;
window.logout      = logout;
window.fmt         = fmt;
window.fmtInt      = fmtInt;
window.corrStrength = corrStrength;
window.litPipeline = litPipeline;
window.animateProgress = animateProgress;
window.show        = show;
window.hide        = hide;
window.buildTable  = buildTable;
window.renderQualityGauge = renderQualityGauge;
