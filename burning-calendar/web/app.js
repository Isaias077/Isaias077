/* Calendario de actividad de quema: mapa, calendario año x mes y gráficos. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("es", { useGrouping: "always" });
const num = (v) => fmt.format(Math.round(v));
const MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
const MONTHS_LONG = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
  "septiembre", "octubre", "noviembre", "diciembre"];
// Rampa secuencial de un solo tono para cantidades; divergente azul-gris-naranja para anomalías.
const SEQ = ["#fbe9df", "#f8cfb9", "#f3ac89", "#ee8a5c", "#e4692f", "#c14f1c", "#983a12", "#6b2609"];
const DIV = ["#184f95", "#3987e5", "#9ec5f4", "#e4e3df", "#f6b89c", "#eb6834", "#7a2c0e"];
const DIV_LABELS = ["÷4 o menos", "÷2", "algo menos", "normal", "algo más", "×2", "×4 o más"];
const EMPTY = "#f1f0ec";

const state = { meta: null, cal: null, mode: "count", year: null, rect: null, overlay: null };

/* ------------------------------------------------------------------ mapa */
const map = L.map("map", { zoomSnap: 0.5 }).setView([-23.5, -58.5], 6);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 14, attribution: "&copy; OpenStreetMap",
}).addTo(map);

function getBbox() {
  return ["w", "s", "e", "n"].map((k) => parseFloat($(k).value));
}

function setBbox(b, fit = true) {
  ["w", "s", "e", "n"].forEach((k, i) => { $(k).value = (+b[i]).toFixed(3); });
  const bounds = [[b[1], b[0]], [b[3], b[2]]];
  if (state.rect) state.rect.setBounds(bounds);
  else state.rect = L.rectangle(bounds, { color: "#1c5cab", weight: 2, fill: false }).addTo(map);
  if (fit) map.fitBounds(bounds, { padding: [20, 20] });
}

// Dibujo del rectángulo: clic y arrastre con el botón "Dibujar área" activo.
let drawing = false;
let corner = null;
$("draw").addEventListener("click", () => {
  drawing = !drawing;
  $("draw").setAttribute("aria-pressed", drawing);
  $("map").classList.toggle("drawing", drawing);
  if (drawing) { map.dragging.disable(); status("Haz clic y arrastra sobre el mapa."); }
  else map.dragging.enable();
});
map.on("mousedown", (ev) => { if (drawing) corner = ev.latlng; });
map.on("mousemove", (ev) => {
  if (!drawing || !corner) return;
  setBbox([Math.min(corner.lng, ev.latlng.lng), Math.min(corner.lat, ev.latlng.lat),
    Math.max(corner.lng, ev.latlng.lng), Math.max(corner.lat, ev.latlng.lat)], false);
});
map.on("mouseup", () => {
  if (!drawing || !corner) return;
  corner = null;
  $("draw").click();
  analyze();
});
["w", "s", "e", "n"].forEach((k) => $(k).addEventListener("change", () => {
  const b = getBbox();
  if (b.every(Number.isFinite) && b[0] < b[2] && b[1] < b[3]) setBbox(b);
}));

/* ------------------------------------------------------------------ utilidades */
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

async function api(path, params) {
  const res = await fetch(`${path}?${new URLSearchParams(params)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Error ${res.status}`);
  }
  return res.json();
}

function query() {
  return { country: $("country").value, bbox: getBbox().join(",") };
}

const tip = $("tip");
function showTip(ev, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  const x = Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
  const y = Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8);
  tip.style.left = `${x}px`;
  tip.style.top = `${y}px`;
}
function hideTip() { tip.hidden = true; }

function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
  if (text !== undefined) el.textContent = text;
  return el;
}

function quantileBreaks(values, n) {
  const v = values.filter((x) => x > 0).sort((a, b) => a - b);
  if (!v.length) return [];
  return Array.from({ length: n - 1 }, (_, i) => v[Math.floor(((i + 1) / n) * (v.length - 1))]);
}
const classOf = (value, breaks) => breaks.filter((b) => value > b).length;
// Clase de anomalía: cociente frente a la mediana, en escalones de ×2.
function anomalyClass(ratio) {
  const l = Math.log2(ratio);
  return l <= -2 ? 0 : l <= -1 ? 1 : l <= -0.32 ? 2 : l < 0.32 ? 3 : l < 1 ? 4 : l < 2 ? 5 : 6;
}
function anomalyText(ratio) {
  if (ratio >= 1.25) return `×${ratio.toFixed(1)} de lo habitual`;
  if (ratio <= 0.8) return `÷${(1 / ratio).toFixed(1)} de lo habitual`;
  return "dentro de lo habitual";
}

/* ------------------------------------------------------------------ indicadores */
function renderKpis(cal, dens) {
  const last = cal.ultimo_mes;
  const peak = cal.climatologia.reduce((a, b) => (b.mediana > a.mediana ? b : a));
  const top = cal.anual.filter((a) => a.completo).reduce((a, b) => (b.armonizado > a.armonizado ? b : a));
  const items = [
    ["Hotspots en el rango elegido", num(dens.total),
      `detectados por ${dens.sensor}, ${$("start").value} a ${$("end").value}`],
    [`Último mes con datos: ${MONTHS_LONG[last.mes_del_anio - 1]} de ${last.anio}`,
      num(last.armonizado),
      `${anomalyText(last.anomalia)}; mediana histórica ${num(last.mediana_historica)}`],
    ["Mes de más actividad", MONTHS_LONG[peak.mes_del_anio - 1],
      `mediana de ${num(peak.mediana)} hotspots`],
    ["Año de más actividad", String(top.anio), `${num(top.armonizado)} hotspots armonizados`],
  ];
  if (cal.tendencia) {
    const t = cal.tendencia.pct_por_decada;
    items.push(["Tendencia de 2001 en adelante", `${t > 0 ? "+" : ""}${t.toFixed(0)} %`,
      "por década, sobre años completos"]);
  }
  $("kpis").replaceChildren(...items.map(([label, value, sub]) => {
    const d = document.createElement("div");
    d.className = "kpi";
    d.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div>` +
      `<div class="sub">${sub}</div>`;
    return d;
  }));
}

/* ------------------------------------------------------------------ calendario */
function renderCalendar() {
  const cal = state.cal;
  const years = cal.anual.map((a) => a.anio);
  const cw = 46, ch = 21, left = 46, top = 22, barW = 150;
  const W = left + 12 * cw + 14 + barW + 60, H = top + years.length * ch + 6;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": "Calendario de hotspots por mes y por año" });
  const breaks = quantileBreaks(cal.meses.map((m) => m.armonizado), SEQ.length);
  MONTHS.forEach((m, i) => svg.append(svgEl("text",
    { x: left + i * cw + cw / 2, y: 14, "text-anchor": "middle" }, m)));
  svg.append(svgEl("text", { x: left + 12 * cw + 14, y: 14 }, "total del año"));

  const maxYear = Math.max(...cal.anual.map((a) => a.armonizado), 1);
  years.forEach((year, r) => {
    const y = top + r * ch;
    const label = svgEl("text", { x: left - 8, y: y + ch / 2 + 4, "text-anchor": "end",
      class: `year${year === state.year ? " sel" : ""}`, tabindex: 0, role: "button",
      "aria-label": `Comparar ${year} en el perfil estacional` }, year);
    const pick = () => { state.year = year; renderCalendar(); renderSeason(); };
    label.addEventListener("click", pick);
    label.addEventListener("keydown", (ev) => { if (ev.key === "Enter") pick(); });
    svg.append(label);
    const a = cal.anual[r];
    svg.append(svgEl("rect", { x: left + 12 * cw + 14, y: y + 3, height: ch - 6, rx: 3,
      width: Math.max(1, (a.armonizado / maxYear) * barW), fill: a.completo ? "#9c9b95" : "#d6d5cf" }));
    svg.append(svgEl("text", { x: left + 12 * cw + 20 + (a.armonizado / maxYear) * barW,
      y: y + ch / 2 + 4 }, num(a.armonizado) + (a.completo ? "" : " (parcial)")));
  });

  cal.meses.forEach((m) => {
    const r = years.indexOf(m.anio), c = m.mes_del_anio - 1;
    const fill = state.mode === "count"
      ? (m.armonizado > 0 ? SEQ[classOf(m.armonizado, breaks)] : EMPTY)
      : DIV[anomalyClass(m.anomalia)];
    const what = `${MONTHS_LONG[c]} de ${m.anio}: ${num(m.armonizado)} hotspots`;
    const cell = svgEl("rect", { x: left + c * cw, y: top + r * ch, width: cw, height: ch, rx: 4,
      fill, class: "cell", tabindex: 0, role: "button", "aria-label": `${what}, ${anomalyText(m.anomalia)}` });
    const detail = m.origen === "VIIRS"
      ? `VIIRS observado. MODIS vio ${num(m.modis_terra + m.modis_aqua)}.`
      : `MODIS convertido (vio ${num(m.modis_terra + m.modis_aqua)}).<br>Rango probable: ${num(m.bajo)} a ${num(m.alto)}.`;
    cell.addEventListener("mousemove", (ev) => showTip(ev,
      `<b>${what}</b><br>${anomalyText(m.anomalia)} (percentil ${m.percentil.toFixed(0)})<br>${detail}`));
    cell.addEventListener("mouseleave", hideTip);
    const zoom = () => {
      const lastDay = new Date(m.anio, c + 1, 0).getDate();
      $("start").value = `${m.mes}-01`;
      $("end").value = `${m.mes}-${String(lastDay).padStart(2, "0")}`;
      refreshMap();
    };
    cell.addEventListener("click", zoom);
    cell.addEventListener("keydown", (ev) => { if (ev.key === "Enter") zoom(); });
    svg.append(cell);
  });

  // Línea que separa la era de MODIS convertido de la de VIIRS observado.
  const r2012 = years.indexOf(2012);
  if (r2012 > 0) {
    const y = top + r2012 * ch;
    svg.append(svgEl("line", { x1: left, x2: left + 12 * cw, y1: y, y2: y, stroke: "#0b0b0b",
      "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  }
  $("calendar").replaceChildren(svg);

  const ramp = state.mode === "count" ? SEQ : DIV;
  const labels = state.mode === "count"
    ? ["menos", "más hotspots"] : ["menos de lo habitual", "más de lo habitual"];
  $("legend").innerHTML = `<span>${labels[0]}</span><span class="ramp">` +
    ramp.map((c, i) => `<span style="background:${c}" title="${state.mode === "count" ? "" : DIV_LABELS[i]}"></span>`).join("") +
    `</span><span>${labels[1]}</span><span>· Línea punteada: desde 2012 es VIIRS observado; ` +
    `antes, MODIS convertido. Clic en un mes para verlo en el mapa.</span>`;
  $("cal-note").textContent = state.mode === "count"
    ? "Cada celda es un mes. El color indica la cantidad de hotspots equivalentes VIIRS."
    : "Cada celda compara el mes con la mediana del mismo mes en los demás años.";
}

/* ------------------------------------------------------------------ perfil estacional */
function renderSeason() {
  const cal = state.cal;
  const W = 520, H = 280, m = { l: 52, r: 16, t: 14, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": "Perfil estacional de hotspots" });
  const yearRows = cal.meses.filter((r) => r.anio === state.year);
  const top = Math.max(...cal.climatologia.map((c) => c.p75), ...yearRows.map((r) => r.armonizado), 1);
  const x = (i) => m.l + (i * (W - m.l - m.r)) / 11;
  const y = (v) => H - m.b - (v / top) * (H - m.t - m.b);
  for (let k = 0; k <= 4; k++) {
    const v = (top * k) / 4;
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "#e4e3df" }));
    svg.append(svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, num(v)));
  }
  MONTHS.forEach((name, i) => svg.append(svgEl("text",
    { x: x(i), y: H - 10, "text-anchor": "middle" }, name)));
  const c = cal.climatologia;
  const band = c.map((d, i) => `${x(i)},${y(d.p75)}`).concat(
    c.map((d, i) => `${x(i)},${y(d.p25)}`).reverse()).join(" ");
  svg.append(svgEl("polygon", { points: band, fill: "#9c9b95", opacity: 0.3 }));
  svg.append(svgEl("polyline", { points: c.map((d, i) => `${x(i)},${y(d.mediana)}`).join(" "),
    fill: "none", stroke: "#52514e", "stroke-width": 2 }));
  if (yearRows.length) {
    svg.append(svgEl("polyline", { fill: "none", stroke: "#eb6834", "stroke-width": 2.5,
      points: yearRows.map((r) => `${x(r.mes_del_anio - 1)},${y(r.armonizado)}`).join(" ") }));
    yearRows.forEach((r) => {
      const dot = svgEl("circle", { cx: x(r.mes_del_anio - 1), cy: y(r.armonizado), r: 5,
        fill: "#eb6834", stroke: "#fff", "stroke-width": 2 });
      dot.addEventListener("mousemove", (ev) => showTip(ev,
        `<b>${MONTHS_LONG[r.mes_del_anio - 1]} de ${r.anio}</b><br>${num(r.armonizado)} hotspots<br>` +
        `mediana histórica: ${num(c[r.mes_del_anio - 1].mediana)}`));
      dot.addEventListener("mouseleave", hideTip);
      svg.append(dot);
    });
  }
  svg.append(svgEl("line", { x1: m.l + 4, x2: m.l + 26, y1: m.t + 4, y2: m.t + 4, stroke: "#eb6834", "stroke-width": 2.5 }));
  svg.append(svgEl("text", { x: m.l + 32, y: m.t + 8 }, String(state.year)));
  svg.append(svgEl("line", { x1: m.l + 84, x2: m.l + 106, y1: m.t + 4, y2: m.t + 4, stroke: "#52514e", "stroke-width": 2 }));
  svg.append(svgEl("text", { x: m.l + 112, y: m.t + 8 }, "mediana de todos los años"));
  $("season").replaceChildren(svg);
  $("season-title").textContent = `Perfil estacional: ${state.year} frente a lo habitual`;
}

/* ------------------------------------------------------------------ armonización */
function renderScatter() {
  const { armonizacion: h, solape } = state.cal;
  const W = 520, H = 280, m = { l: 52, r: 16, t: 14, b: 38 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": "Conteos mensuales de MODIS frente a VIIRS en los años en que conviven" });
  const top = Math.log10(Math.max(...solape.map((p) => Math.max(p.modis, p.viirs)), 10) + 1) * 1.05;
  const sx = (v) => m.l + (Math.log10(v + 1) / top) * (W - m.l - m.r);
  const sy = (v) => H - m.b - (Math.log10(v + 1) / top) * (H - m.t - m.b);
  for (let p = 0; p <= Math.floor(top); p++) {
    const v = 10 ** p - (p === 0 ? 1 : 0);
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), stroke: "#e4e3df" }));
    svg.append(svgEl("text", { x: m.l - 6, y: sy(v) + 4, "text-anchor": "end" }, num(v)));
    svg.append(svgEl("text", { x: sx(v), y: H - m.b + 16, "text-anchor": "middle" }, num(v)));
  }
  svg.append(svgEl("text", { x: (W + m.l) / 2, y: H - 4, "text-anchor": "middle" }, "hotspots MODIS en el mes"));
  svg.append(svgEl("text", { x: 12, y: H / 2, "text-anchor": "middle",
    transform: `rotate(-90 12 ${H / 2})` }, "hotspots VIIRS en el mes"));
  const lim = 10 ** top - 1;
  svg.append(svgEl("line", { x1: sx(0), y1: sy(0), x2: sx(lim), y2: sy(lim), stroke: "#9c9b95",
    "stroke-dasharray": "4 4" }));
  solape.forEach((p) => {
    const dot = svgEl("circle", { cx: sx(p.modis), cy: sy(p.viirs), r: 4, fill: "#2a78d6",
      opacity: 0.7, stroke: "#fff", "stroke-width": 1 });
    dot.addEventListener("mousemove", (ev) => showTip(ev,
      `<b>${p.mes}</b><br>MODIS: ${num(p.modis)}<br>VIIRS: ${num(p.viirs)}`));
    dot.addEventListener("mouseleave", hideTip);
    svg.append(dot);
  });
  const curve = [];
  for (let i = 0; i <= 40; i++) {
    const v = 10 ** ((top * i) / 40) - 1;
    curve.push(`${sx(v)},${sy(Math.max(Math.exp(h.a + h.b * Math.log1p(v)) - 1, 0))}`);
  }
  svg.append(svgEl("polyline", { points: curve.join(" "), fill: "none", stroke: "#eb6834", "stroke-width": 2.5 }));
  $("scatter").replaceChildren(svg);

  const where = h.origen === "area" ? "en esta área"
    : "de todo el país, porque esta área tiene pocos meses con fuego";
  let text = `Cada punto es un mes entre ${h.viirs_desde} y ${h.viirs_hasta}, cuando conviven los dos ` +
    `sensores. La curva naranja es el ajuste ${where}; la línea gris sería "ven lo mismo".`;
  if (h.validacion) {
    const v = h.validacion;
    text += ` Validación dejando un año afuera: sin armonizar, MODIS difiere de VIIRS por un factor ` +
      `${v.factor_mensual_sin_armonizar.toFixed(1)} en un mes típico; armonizado, por ${v.factor_mensual.toFixed(2)} ` +
      `(${v.factor_anual.toFixed(2)} en el total anual).`;
  }
  $("harm-note").textContent = text;
}

/* ------------------------------------------------------------------ mapa de densidad */
function drawDensity(d, bbox) {
  const canvas = document.createElement("canvas");
  canvas.width = d.nx;
  canvas.height = d.ny;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(d.nx, d.ny);
  const top = Math.log1p(d.max || 1);
  for (let iy = 0; iy < d.ny; iy++) {
    for (let ix = 0; ix < d.nx; ix++) {
      const v = d.counts[iy * d.nx + ix];
      if (!v) continue;
      const k = Math.min(SEQ.length - 1, 2 + Math.floor((Math.log1p(v) / top) * (SEQ.length - 2)));
      const hex = SEQ[k];
      const o = ((d.ny - 1 - iy) * d.nx + ix) * 4;
      img.data[o] = parseInt(hex.slice(1, 3), 16);
      img.data[o + 1] = parseInt(hex.slice(3, 5), 16);
      img.data[o + 2] = parseInt(hex.slice(5, 7), 16);
      img.data[o + 3] = 235;
    }
  }
  ctx.putImageData(img, 0, 0);
  if (state.overlay) state.overlay.remove();
  state.overlay = L.imageOverlay(canvas.toDataURL(), [[bbox[1], bbox[0]], [bbox[3], bbox[2]]],
    { opacity: 0.9, className: "density" }).addTo(map);
  state.overlay.getElement().style.imageRendering = "pixelated";
}

async function refreshMap() {
  const q = { ...query(), start: $("start").value, end: $("end").value };
  const dens = await api("/api/density", q);
  drawDensity(dens, getBbox());
  $("dl-hotspots").href = `/api/hotspots.csv?${new URLSearchParams(q)}`;
  if (state.cal) renderKpis(state.cal, dens);
  status(`${num(dens.total)} hotspots de ${dens.sensor} en el mapa, del ${q.start} al ${q.end}.`);
  return dens;
}

/* ------------------------------------------------------------------ flujo principal */
async function analyze() {
  const b = getBbox();
  if (!b.every(Number.isFinite) || b[0] >= b[2] || b[1] >= b[3]) {
    status("El área no es válida: revisa que oeste < este y sur < norte.", true);
    return;
  }
  if ($("start").value > $("end").value) {
    status("La fecha inicial es posterior a la final.", true);
    return;
  }
  setBbox(b, false);
  status("Calculando…");
  try {
    state.cal = await api("/api/calendar", query());
    const complete = state.cal.anual.filter((a) => a.completo);
    if (!state.year) state.year = complete[complete.length - 1].anio;
    $("dl-calendar").href = `/api/calendar.csv?${new URLSearchParams(query())}`;
    renderCalendar();
    renderSeason();
    renderScatter();
    await refreshMap();
  } catch (err) {
    status(err.message, true);
  }
}

function setMode(mode) {
  state.mode = mode;
  $("mode-count").setAttribute("aria-pressed", mode === "count");
  $("mode-anom").setAttribute("aria-pressed", mode === "anom");
  if (state.cal) renderCalendar();
}
$("mode-count").addEventListener("click", () => setMode("count"));
$("mode-anom").addEventListener("click", () => setMode("anom"));
$("go").addEventListener("click", analyze);
$("whole").addEventListener("click", () => { setBbox(currentCountry().bbox); analyze(); });
$("dl-svg").addEventListener("click", () => {
  const svg = $("calendar").querySelector("svg");
  if (!svg) return;
  const blob = new Blob([new XMLSerializer().serializeToString(svg)], { type: "image/svg+xml" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "calendario_de_quema.svg";
  a.click();
  URL.revokeObjectURL(a.href);
});

function currentCountry() {
  return state.meta.paises.find((p) => p.pais === $("country").value);
}

function selectCountry() {
  const c = currentCountry();
  setBbox(c.bbox);
  const end = new Date(c.ultimo_dia);
  const start = new Date(end);
  start.setFullYear(end.getFullYear() - 1);
  start.setDate(start.getDate() + 1);
  $("end").value = c.ultimo_dia;
  $("start").value = start.toISOString().slice(0, 10);
  $("start").min = $("end").min = c.primer_dia;
  $("start").max = $("end").max = c.ultimo_dia;
  state.year = null;
  analyze();
}
$("country").addEventListener("change", selectCountry);

(async function init() {
  try {
    state.meta = await api("/api/meta", {});
    if (!state.meta.paises.length) {
      status("No hay datos cargados. Corre: python -m server.ingest --country Paraguay", true);
      return;
    }
    $("country").replaceChildren(...state.meta.paises.map((p) => new Option(p.pais, p.pais)));
    selectCountry();
  } catch (err) {
    status(err.message, true);
  }
})();
