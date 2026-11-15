/* Calendario de actividad de quema: cualquier país, provincia o rectángulo, con datos a demanda. */
"use strict";

const $ = (id) => document.getElementById(id);
const fmt = new Intl.NumberFormat("es", { useGrouping: "always" });
const num = (v) => (v === null || v === undefined ? "—" : fmt.format(Math.round(v)));
const MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"];
const MONTHS_LONG = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
  "septiembre", "octubre", "noviembre", "diciembre"];
// Rampa secuencial de un solo tono para cantidades; divergente azul-gris-naranja para anomalías.
const SEQ = ["#fbe9df", "#f8cfb9", "#f3ac89", "#ee8a5c", "#e4692f", "#c14f1c", "#983a12", "#6b2609"];
const DIV = ["#184f95", "#3987e5", "#9ec5f4", "#e4e3df", "#f6b89c", "#eb6834", "#7a2c0e"];
const EMPTY = "#f1f0ec";
const PENDING = "#d6d5cf";

const state = {
  countries: [], area: null, cal: null, mode: "count", year: null, datesTouched: false,
  outline: null, rect: null, overlay: null, poll: null, lastProgress: "",
};

/* ------------------------------------------------------------------ utilidades */
function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}

async function api(path, params = {}) {
  const res = await fetch(`${path}?${new URLSearchParams(params)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `Error ${res.status}`);
  }
  return res.json();
}

const tip = $("tip");
function showTip(ev, html) {
  tip.innerHTML = html;
  tip.hidden = false;
  tip.style.left = `${Math.min(ev.clientX + 14, window.innerWidth - tip.offsetWidth - 8)}px`;
  tip.style.top = `${Math.min(ev.clientY + 14, window.innerHeight - tip.offsetHeight - 8)}px`;
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
function anomalyClass(ratio) {
  const l = Math.log2(ratio);
  return l <= -2 ? 0 : l <= -1 ? 1 : l <= -0.32 ? 2 : l < 0.32 ? 3 : l < 1 ? 4 : l < 2 ? 5 : 6;
}
function anomalyText(ratio) {
  if (ratio === null || ratio === undefined) return "sin datos todavía";
  if (ratio >= 1.25) return `×${ratio.toFixed(1)} de lo habitual`;
  if (ratio <= 0.8) return `÷${(1 / ratio).toFixed(1)} de lo habitual`;
  return "dentro de lo habitual";
}

/* ------------------------------------------------------------------ área elegida */
function areaParams() {
  const a = state.area;
  if (!a) return null;
  if (a.kind === "rect") return { bbox: a.bbox.join(",") };
  if (a.province) return { provincia: a.province };
  return { pais: a.country };
}

function saveHash() {
  const p = areaParams();
  history.replaceState(null, "", p ? `#${new URLSearchParams(p)}` : "#");
}

/* ------------------------------------------------------------------ mapa */
const map = L.map("map", { zoomSnap: 0.5, worldCopyJump: true }).setView([-15, -60], 3);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  maxZoom: 14, attribution: "&copy; OpenStreetMap",
}).addTo(map);

function showOutline(geom, bbox) {
  if (state.outline) state.outline.remove();
  if (state.rect) { state.rect.remove(); state.rect = null; }
  state.outline = L.geoJSON(geom, { style: { color: "#1c5cab", weight: 2, fill: false }, interactive: false }).addTo(map);
  map.fitBounds([[bbox[1], bbox[0]], [bbox[3], bbox[2]]], { padding: [16, 16] });
}

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
  const b = [[corner.lat, corner.lng], [ev.latlng.lat, ev.latlng.lng]];
  if (state.rect) state.rect.setBounds(b);
  else state.rect = L.rectangle(b, { color: "#1c5cab", weight: 2, fill: false, interactive: false }).addTo(map);
});
map.on("mouseup", (ev) => {
  if (!drawing || !corner) return;
  const bbox = [Math.min(corner.lng, ev.latlng.lng), Math.min(corner.lat, ev.latlng.lat),
    Math.max(corner.lng, ev.latlng.lng), Math.max(corner.lat, ev.latlng.lat)].map((v) => +v.toFixed(3));
  corner = null;
  $("draw").click();
  if (bbox[2] - bbox[0] < 0.05 || bbox[3] - bbox[1] < 0.05) { status("El rectángulo es muy chico."); return; }
  if (state.outline) { state.outline.remove(); state.outline = null; }
  $("country").value = "";
  $("province").replaceChildren(new Option("Todo el país", ""));
  $("province").disabled = true;
  selectArea({ kind: "rect", bbox });
});
map.on("click", async (ev) => {
  if (drawing || ev.originalEvent.defaultPrevented) return;
  try {
    const hit = await api("/api/ubicar", { lon: ev.latlng.lng, lat: ev.latlng.lat });
    await chooseCountry(hit.pais, false);
    if (hit.provincia) {
      $("province").value = hit.provincia;
      await chooseProvince(hit.provincia);
    } else {
      selectArea({ kind: "country", country: hit.pais });
    }
  } catch (err) { status(err.message, true); }
});

/* ------------------------------------------------------------------ selectores */
function countryByName(text) {
  const t = text.trim().toLowerCase();
  return state.countries.find((c) => c.nombre.toLowerCase() === t)
    || state.countries.find((c) => c.iso3.toLowerCase() === t);
}

async function chooseCountry(iso3, analyze = true) {
  const info = await api(`/api/paises/${iso3}`);
  $("country").value = info.nombre;
  const sel = $("province");
  sel.replaceChildren(new Option("Todo el país", ""), ...info.provincias.map((p) => new Option(p.nombre, p.id)));
  sel.disabled = false;
  showOutline(info.geom, info.bbox);
  if (analyze) selectArea({ kind: "country", country: iso3 });
  return info;
}

async function chooseProvince(pid) {
  const p = await api(`/api/provincias/${pid}`);
  showOutline(p.geom, p.bbox);
  selectArea({ kind: "province", country: p.pais, province: pid });
}

$("country").addEventListener("change", () => {
  const c = countryByName($("country").value);
  if (c) chooseCountry(c.iso3).catch((err) => status(err.message, true));
  else status("No encuentro ese país en la lista.", true);
});
$("province").addEventListener("change", () => {
  const pid = $("province").value;
  const c = countryByName($("country").value);
  if (pid) chooseProvince(pid).catch((err) => status(err.message, true));
  else if (c) chooseCountry(c.iso3).catch((err) => status(err.message, true));
});

/* ------------------------------------------------------------------ flujo principal */
function selectArea(area) {
  state.area = area;
  state.year = null;
  state.lastProgress = "";
  saveHash();
  $("empty").hidden = true;
  load();
}

const sameArea = (params) => JSON.stringify(areaParams()) === JSON.stringify(params);

async function load() {
  const params = areaParams();
  if (!params) return;
  status("Calculando…");
  try {
    const cal = await api("/api/calendario", params);
    if (!sameArea(params)) return;               // mientras tanto se eligió otra área
    state.cal = cal;
    renderAll();
    await refreshMap();
    watchProgress(cal.estado);
  } catch (err) {
    status(err.message, true);
  }
}

function progressKey(est) {
  return est.paises.map((p) => `${p.pais}:${p.archivos_listos}:${p.reciente}:${p.fase}`).join("|");
}

function watchProgress(est) {
  renderProgress(est);
  clearTimeout(state.poll);
  if (!est.en_curso) return;
  state.lastProgress = progressKey(est);
  const params = areaParams();
  state.poll = setTimeout(async () => {
    if (!sameArea(params)) return;
    try {
      const now = await api("/api/estado", params);
      if (progressKey(now) !== state.lastProgress) load();   // llegaron datos nuevos: recalcular
      else watchProgress(now);
    } catch (err) { status(err.message, true); }
  }, 3000);
}

function renderProgress(est) {
  const box = $("progress");
  const busy = est.paises.filter((p) => p.en_curso || p.sin_archivo || p.error);
  box.hidden = !busy.length;
  box.replaceChildren(...busy.map((p) => {
    const div = document.createElement("div");
    div.className = "prog";
    let text, pct = null;
    if (p.error) text = `${p.nombre}: no se pudo completar la descarga (${p.error}). Se muestra lo que llegó.`;
    else if (p.sin_archivo && !p.en_curso) text = `${p.nombre}: FIRMS no tiene archivo histórico con ese nombre de país; solo hay año en curso.`;
    else if (!p.fase || p.fase === "iniciando" || p.fase.startsWith("buscando")) text = `${p.nombre}: buscando el país en FIRMS…`;
    else {
      const files = p.archivos_total ? `${p.archivos_listos} de ${p.archivos_total} archivos anuales` : "archivo histórico listo";
      const rec = p.reciente && p.reciente !== "listo" ? ` · año en curso: ${p.reciente} pedidos` : "";
      text = `Descargando ${p.nombre}: ${files}${rec}. El calendario se completa solo.`;
      if (p.archivos_total) pct = p.archivos_listos / p.archivos_total;
    }
    div.innerHTML = `<div>${text}</div>` + (pct !== null
      ? `<div class="bar" role="progressbar" aria-valuenow="${Math.round(pct * 100)}" aria-valuemin="0" aria-valuemax="100"><span style="width:${(pct * 100).toFixed(1)}%"></span></div>` : "");
    return div;
  }));
}

function renderAll() {
  const cal = state.cal;
  $("results").hidden = false;
  $("area-title").textContent = cal.area.nombre;
  const cov = cal.cobertura;
  let text = cov.modis_hasta
    ? `Datos hasta el ${cov.modis_hasta} (MODIS) y el ${cov.viirs_hasta} (VIIRS S-NPP).`
    : "Todavía no llegó ningún dato de esta área: se están descargando.";
  if (cov.actualizado) text += ` Año en curso actualizado el ${cov.actualizado.slice(0, 10)}.`;
  $("coverage").textContent = text;
  if (!state.datesTouched && cov.modis_hasta) {
    const end = new Date(cov.modis_hasta);
    const start = new Date(end);
    start.setFullYear(end.getFullYear() - 1);
    start.setDate(start.getDate() + 1);
    $("end").value = cov.modis_hasta;
    $("start").value = start.toISOString().slice(0, 10);
  }
  if (!state.year) state.year = cal.ultimo_mes.anio;
  $("dl-calendar").href = `/api/calendario.csv?${new URLSearchParams(areaParams())}`;
  renderKpis();
  renderCalendar();
  renderSeason();
  renderScatter();
}

/* ------------------------------------------------------------------ indicadores */
function renderKpis(dens) {
  const cal = state.cal;
  const last = cal.ultimo_mes;
  const known = cal.climatologia.filter((c) => c.mediana !== null);
  const peak = known.length ? known.reduce((a, b) => (b.mediana > a.mediana ? b : a)) : null;
  const full = cal.anual.filter((a) => a.completo && a.armonizado !== null);
  const top = full.length ? full.reduce((a, b) => (b.armonizado > a.armonizado ? b : a)) : null;
  const items = [];
  if (dens) items.push(["Hotspots en el rango elegido", num(dens.total), `detectados por ${dens.sensor}, ${$("start").value} a ${$("end").value}`]);
  items.push(!last.hay_datos ? ["Último mes", "—", "todavía descargando"] : last.parcial
    ? [`Mes en curso: 1 al ${last.dias} de ${MONTHS_LONG[last.mes_del_anio - 1]} de ${last.anio}`, num(last.armonizado),
      `${anomalyText(last.anomalia)}; mediana de los mismos días en otros años: ${num(last.mediana_historica)}`]
    : [`Último mes: ${MONTHS_LONG[last.mes_del_anio - 1]} de ${last.anio}`, num(last.armonizado),
      `${anomalyText(last.anomalia)}; mediana histórica ${num(last.mediana_historica)}`]);
  if (peak) items.push(["Mes de más actividad", MONTHS_LONG[peak.mes_del_anio - 1], `mediana de ${num(peak.mediana)} hotspots`]);
  if (top) items.push(["Año de más actividad", String(top.anio), `${num(top.armonizado)} hotspots armonizados`]);
  if (cal.tendencia) {
    const t = cal.tendencia.pct_por_decada;
    items.push([`Tendencia desde ${cal.tendencia.desde}`, `${t > 0 ? "+" : ""}${t.toFixed(0)} %`, "por década, sobre años completos"]);
  }
  $("kpis").replaceChildren(...items.map(([label, value, sub]) => {
    const d = document.createElement("div");
    d.className = "kpi";
    d.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub}</div>`;
    return d;
  }));
}

/* ------------------------------------------------------------------ calendario */
function renderCalendar() {
  const cal = state.cal;
  const years = cal.anual.map((a) => a.anio);
  const cw = 46, ch = 21, left = 46, top = 22, barW = 150;
  const W = left + 12 * cw + 14 + barW + 90, H = top + years.length * ch + 6;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img",
    "aria-label": "Calendario de hotspots por mes y por año" });
  const defs = svgEl("defs");
  const pat = svgEl("pattern", { id: "stripes", width: 6, height: 6, patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)" });
  pat.append(svgEl("rect", { width: 2.5, height: 6, fill: "#ffffff", opacity: 0.75 }));
  defs.append(pat);
  svg.append(defs);
  const breaks = quantileBreaks(cal.meses.map((m) => m.armonizado || 0), SEQ.length);
  MONTHS.forEach((m, i) => svg.append(svgEl("text", { x: left + i * cw + cw / 2, y: 14, "text-anchor": "middle" }, m)));
  svg.append(svgEl("text", { x: left + 12 * cw + 14, y: 14 }, "total del año"));

  const maxYear = Math.max(...cal.anual.map((a) => a.armonizado || 0), 1);
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
    const v = a.armonizado || 0;
    svg.append(svgEl("rect", { x: left + 12 * cw + 14, y: y + 3, height: ch - 6, rx: 3,
      width: Math.max(1, (v / maxYear) * barW), fill: a.completo ? "#9c9b95" : "#d6d5cf" }));
    svg.append(svgEl("text", { x: left + 12 * cw + 20 + (v / maxYear) * barW, y: y + ch / 2 + 4 },
      a.armonizado === null ? "descargando…" : num(a.armonizado) + (a.completo ? "" : " (parcial)")));
  });

  cal.meses.forEach((m) => {
    const r = years.indexOf(m.anio), c = m.mes_del_anio - 1;
    const pending = m.armonizado === null;
    const fill = pending ? PENDING : state.mode === "count"
      ? (m.armonizado > 0 ? SEQ[classOf(m.armonizado, breaks)] : EMPTY)
      : DIV[anomalyClass(m.anomalia)];
    const days = m.parcial ? `1 al ${cal.ultimo_mes.dias} de ` : "";
    const what = pending ? `${MONTHS_LONG[c]} de ${m.anio}: todavía descargando`
      : `${days}${MONTHS_LONG[c]} de ${m.anio}: ${num(m.armonizado)} hotspots`;
    const x = left + c * cw, y = top + r * ch;
    const cell = svgEl("rect", { x, y, width: cw, height: ch, rx: 4, fill, class: "cell", tabindex: 0,
      role: "button", "aria-label": pending ? what : `${what}, ${anomalyText(m.anomalia)}` });
    let detail = "";
    if (!pending) {
      detail = m.origen === "VIIRS"
        ? `VIIRS observado. MODIS vio ${num(m.modis_terra + m.modis_aqua)}.`
        : `MODIS convertido (vio ${num(m.modis_terra + m.modis_aqua)}).<br>Rango probable: ${num(m.bajo)} a ${num(m.alto)}.`;
      if (m.parcial) detail += "<br><i>Mes en curso: comparado con los mismos días de otros años.</i>";
    }
    cell.addEventListener("mousemove", (ev) => showTip(ev, pending ? `<b>${what}</b>`
      : `<b>${what}</b><br>${anomalyText(m.anomalia)} (percentil ${m.percentil.toFixed(0)})<br>${detail}`));
    cell.addEventListener("mouseleave", hideTip);
    const zoom = () => {
      const lastDay = new Date(m.anio, c + 1, 0).getDate();
      $("start").value = `${m.mes}-01`;
      $("end").value = `${m.mes}-${String(lastDay).padStart(2, "0")}`;
      state.datesTouched = true;
      refreshMap().catch((e) => status(e.message, true));
    };
    cell.addEventListener("click", zoom);
    cell.addEventListener("keydown", (ev) => { if (ev.key === "Enter") zoom(); });
    svg.append(cell);
    if (m.parcial || pending) {
      svg.append(svgEl("rect", { x: x + 1, y: y + 1, width: cw - 2, height: ch - 2, rx: 3,
        fill: "url(#stripes)", "pointer-events": "none" }));
    }
  });

  const r2012 = years.indexOf(2012);
  if (r2012 > 0) {
    const y = top + r2012 * ch;
    svg.append(svgEl("line", { x1: left, x2: left + 12 * cw, y1: y, y2: y, stroke: "#0b0b0b",
      "stroke-width": 1.5, "stroke-dasharray": "5 4" }));
  }
  $("calendar").replaceChildren(svg);

  const ramp = state.mode === "count" ? SEQ : DIV;
  const labels = state.mode === "count" ? ["menos", "más hotspots"] : ["menos de lo habitual", "más de lo habitual"];
  $("legend").innerHTML = `<span>${labels[0]}</span><span class="ramp">` +
    ramp.map((c) => `<span style="background:${c}"></span>`).join("") +
    `</span><span>${labels[1]}</span><span>· Rayado: mes en curso o todavía descargando. ` +
    `Línea punteada: desde 2012 es VIIRS observado; antes, MODIS convertido. Clic en un mes para verlo en el mapa.</span>`;
  $("cal-note").textContent = state.mode === "count"
    ? "Cada celda es un mes. El color indica la cantidad de hotspots equivalentes VIIRS."
    : "Cada celda compara el mes con la mediana del mismo mes en los demás años.";
}

/* ------------------------------------------------------------------ perfil estacional */
function renderSeason() {
  const cal = state.cal;
  const W = 520, H = 280, m = { l: 52, r: 16, t: 14, b: 30 };
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": "Perfil estacional" });
  const yearRows = cal.meses.filter((r) => r.anio === state.year && r.armonizado !== null);
  const c = cal.climatologia;
  const top = Math.max(...c.map((d) => d.p75 || 0), ...yearRows.map((r) => r.armonizado), 1);
  const x = (i) => m.l + (i * (W - m.l - m.r)) / 11;
  const y = (v) => H - m.b - (v / top) * (H - m.t - m.b);
  for (let k = 0; k <= 4; k++) {
    const v = (top * k) / 4;
    svg.append(svgEl("line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v), stroke: "#e4e3df" }));
    svg.append(svgEl("text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, num(v)));
  }
  MONTHS.forEach((name, i) => svg.append(svgEl("text", { x: x(i), y: H - 10, "text-anchor": "middle" }, name)));
  if (c.every((d) => d.mediana !== null)) {
    const band = c.map((d, i) => `${x(i)},${y(d.p75)}`).concat(c.map((d, i) => `${x(i)},${y(d.p25)}`).reverse()).join(" ");
    svg.append(svgEl("polygon", { points: band, fill: "#9c9b95", opacity: 0.3 }));
    svg.append(svgEl("polyline", { points: c.map((d, i) => `${x(i)},${y(d.mediana)}`).join(" "), fill: "none", stroke: "#52514e", "stroke-width": 2 }));
  }
  if (yearRows.length) {
    svg.append(svgEl("polyline", { fill: "none", stroke: "#eb6834", "stroke-width": 2.5,
      points: yearRows.map((r) => `${x(r.mes_del_anio - 1)},${y(r.armonizado)}`).join(" ") }));
    yearRows.forEach((r) => {
      const dot = svgEl("circle", { cx: x(r.mes_del_anio - 1), cy: y(r.armonizado), r: 5,
        fill: r.parcial ? "#ffffff" : "#eb6834", stroke: r.parcial ? "#eb6834" : "#fff", "stroke-width": 2 });
      dot.addEventListener("mousemove", (ev) => showTip(ev,
        `<b>${MONTHS_LONG[r.mes_del_anio - 1]} de ${r.anio}</b>${r.parcial ? " (mes en curso, incompleto)" : ""}` +
        `<br>${num(r.armonizado)} hotspots<br>mediana histórica: ${num(c[r.mes_del_anio - 1].mediana)}`));
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
  if (!solape.length || h.a_exacto === null) {
    $("scatter").replaceChildren();
    $("harm-note").textContent = "Todavía no hay meses con los dos sensores para armonizar. " +
      "Los meses que solo tienen MODIS aparecen como pendientes hasta que llegue VIIRS.";
    return;
  }
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
  svg.append(svgEl("text", { x: 12, y: H / 2, "text-anchor": "middle", transform: `rotate(-90 12 ${H / 2})` }, "hotspots VIIRS en el mes"));
  const lim = 10 ** top - 1;
  svg.append(svgEl("line", { x1: sx(0), y1: sy(0), x2: sx(lim), y2: sy(lim), stroke: "#9c9b95", "stroke-dasharray": "4 4" }));
  solape.forEach((p) => {
    const dot = svgEl("circle", { cx: sx(p.modis), cy: sy(p.viirs), r: 4, fill: "#2a78d6", opacity: 0.7, stroke: "#fff", "stroke-width": 1 });
    dot.addEventListener("mousemove", (ev) => showTip(ev, `<b>${p.mes}</b><br>MODIS: ${num(p.modis)}<br>VIIRS: ${num(p.viirs)}`));
    dot.addEventListener("mouseleave", hideTip);
    svg.append(dot);
  });
  const curve = [];
  for (let i = 0; i <= 40; i++) {
    const v = 10 ** ((top * i) / 40) - 1;
    curve.push(`${sx(v)},${sy(Math.max(Math.exp(h.a_exacto + h.b * Math.log1p(v)) - 1, 0))}`);
  }
  svg.append(svgEl("polyline", { points: curve.join(" "), fill: "none", stroke: "#eb6834", "stroke-width": 2.5 }));
  $("scatter").replaceChildren(svg);
  const where = h.origen === "area" ? "en esta área"
    : "de los países que la contienen, porque esta área tiene pocos fuegos para un ajuste propio";
  let text = `Cada punto es un mes con datos de los dos sensores. La curva naranja es el ajuste ${where}; ` +
    `la línea gris sería "ven lo mismo".`;
  if (h.validacion) {
    const v = h.validacion;
    text += ` Validación dejando un año afuera: sin armonizar, MODIS difiere de VIIRS por un factor ` +
      `${v.factor_mensual_sin_armonizar.toFixed(1)} en un mes típico; armonizado, por ${v.factor_mensual.toFixed(2)} ` +
      `(${v.factor_anual.toFixed(2)} en el total anual).`;
  }
  $("harm-note").textContent = text;
}

/* ------------------------------------------------------------------ mapa de densidad */
function drawDensity(d) {
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
      const hex = SEQ[Math.min(SEQ.length - 1, 2 + Math.floor((Math.log1p(v) / top) * (SEQ.length - 2)))];
      const o = ((d.ny - 1 - iy) * d.nx + ix) * 4;
      img.data[o] = parseInt(hex.slice(1, 3), 16);
      img.data[o + 1] = parseInt(hex.slice(3, 5), 16);
      img.data[o + 2] = parseInt(hex.slice(5, 7), 16);
      img.data[o + 3] = 235;
    }
  }
  ctx.putImageData(img, 0, 0);
  if (state.overlay) state.overlay.remove();
  const b = d.bbox;
  state.overlay = L.imageOverlay(canvas.toDataURL(), [[b[1], b[0]], [b[3], b[2]]], { opacity: 0.9 }).addTo(map);
  state.overlay.getElement().style.imageRendering = "pixelated";
}

async function refreshMap() {
  const params = areaParams();
  if (!params || !$("start").value || !$("end").value) return;
  if ($("start").value > $("end").value) { status("La fecha inicial es posterior a la final.", true); return; }
  const q = { ...params, inicio: $("start").value, fin: $("end").value };
  const dens = await api("/api/densidad", q);
  if (!sameArea(params)) return;
  drawDensity(dens);
  $("dl-hotspots").href = `/api/hotspots.csv?${new URLSearchParams(q)}`;
  renderKpis(dens);
  const why = dens.sensor === "MODIS" && q.inicio >= "2012-01-20"
    ? ` Se usa MODIS porque VIIRS S-NPP solo tiene datos hasta el ${dens.viirs_hasta}.` : "";
  status(`${num(dens.total)} hotspots de ${dens.sensor} en el mapa, del ${q.inicio} al ${q.fin}.${why}`);
}

$("apply-dates").addEventListener("click", () => { state.datesTouched = true; refreshMap().catch((e) => status(e.message, true)); });

function setMode(mode) {
  state.mode = mode;
  $("mode-count").setAttribute("aria-pressed", mode === "count");
  $("mode-anom").setAttribute("aria-pressed", mode === "anom");
  if (state.cal) renderCalendar();
}
$("mode-count").addEventListener("click", () => setMode("count"));
$("mode-anom").addEventListener("click", () => setMode("anom"));
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

/* ------------------------------------------------------------------ inicio */
async function restoreFromHash() {
  const h = new URLSearchParams(location.hash.slice(1));
  const current = areaParams();
  if (current && new URLSearchParams(current).toString() === h.toString()) return;
  if (h.get("provincia")) {
    const p = await api(`/api/provincias/${h.get("provincia")}`);
    await chooseCountry(p.pais, false);
    $("province").value = p.id;
    await chooseProvince(p.id);
  } else if (h.get("pais")) {
    await chooseCountry(h.get("pais"));
  } else if (h.get("bbox")) {
    const b = h.get("bbox").split(",").map(Number);
    if (state.outline) { state.outline.remove(); state.outline = null; }
    if (state.rect) state.rect.remove();
    state.rect = L.rectangle([[b[1], b[0]], [b[3], b[2]]], { color: "#1c5cab", weight: 2, fill: false, interactive: false }).addTo(map);
    map.fitBounds(state.rect.getBounds(), { padding: [16, 16] });
    $("country").value = "";
    $("province").replaceChildren(new Option("Todo el país", ""));
    $("province").disabled = true;
    selectArea({ kind: "rect", bbox: b });
  }
}

window.addEventListener("hashchange", () => restoreFromHash().catch((err) => status(err.message, true)));

(async function init() {
  try {
    state.countries = (await api("/api/paises")).paises;
    $("country-list").replaceChildren(...state.countries.map((c) => new Option(c.nombre)));
    await restoreFromHash();
  } catch (err) {
    status(err.message, true);
  }
})();
