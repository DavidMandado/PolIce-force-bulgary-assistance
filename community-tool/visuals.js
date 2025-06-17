/* ---------- config ---------- */
const CRIME_CSV = "../data/crime_fixed_data.csv";
const LOOKUP_JSON = "../data/lsoa_to_ward.json";

/* ---------- tiny helpers ---------- */
const $ = (id) => document.getElementById(id);
const showError = (msg) => {
  $("error").textContent = msg;
  $("status").style.display = "none";
};
const clearError = () => ($("error").textContent = "");

const minusMonths = (ym, n) => {
  const [Y, M] = ym.split("-").map(Number);
  const d = new Date(Y, M - 1 - n);
  return d.toISOString().slice(0, 7);
};
function* monthRange(start, end) {
  let [ys, ms] = start.split("-").map(Number);
  const [ye, me] = end.split("-").map(Number);
  while (ys < ye || (ys === ye && ms <= me)) {
    yield `${ys}-${String(ms).padStart(2, "0")}`;
    ms++;
    if (ms > 12) {
      ms = 1;
      ys++;
    }
  }
}

/* ---------- data containers ---------- */
const wardName = new Map(); // Ward→Name
const wardMonth = new Map(); // "YYYY-MM|Ward"→count
const lsoaCnt = new Map(); // "Ward|LSOA"→count
let monthsArr = [],
  londonMean = {};

/* ---------- 0. simple tab switch ---------- */
$("btn-dash").onclick = () => {
  showTab("dash");
};
$("btn-feedback").onclick = () => {
  showTab("feedback");
};

function showTab(tab) {
  $("tab-dash").style.display = tab === "dash" ? "block" : "none";
  $("tab-feedback").style.display = tab === "feedback" ? "block" : "none";
  $("btn-dash").classList.toggle("active", tab === "dash");
  $("btn-feedback").classList.toggle("active", tab === "feedback");
}

/* ---------- 1. load lookup ---------- */
let lsoaToWard = {};
fetch(LOOKUP_JSON)
  .then((res) => res.json())
  .then((json) => {
    Object.entries(json).forEach(([lsoa, ward]) => {
      lsoaToWard[lsoa.trim()] = ward.ward_code; // ← trim here
      if (!wardName.has(ward.ward_code)) {
        wardName.set(ward.ward_code, ward.ward_name);
      }
    });
    if (!wardName.size) return showError("Lookup JSON is empty or malformed.");
    loadCrime();
  })
  .catch((err) => showError("Lookup JSON fetch failed: " + err.message));

/* ---------- 2. load crime CSV (fetch → parse) ---------- */
async function loadCrime() {
  try {
    const resp = await fetch(CRIME_CSV);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const csvText = await resp.text();

    const parseRes = Papa.parse(csvText, {
      header: true,
      skipEmptyLines: true,
      transformHeader: (h) => h.trim().toLowerCase().replace(/\s+/g, "_"), // "month", "lsoa_code"
    });

    const rows = [];
    let maxMonth = null;

    parseRes.data.forEach((r) => {
      const rawMonth = (r.month || "").trim(); // e.g. "2024-3", "2024-03-01"
      const mMatch = /^(\d{4})[-/](\d{1,2})/.exec(rawMonth);
      if (!mMatch) return; // skip bad rows
      const ym = `${mMatch[1]}-${mMatch[2].padStart(2, "0")}`; // → "YYYY-MM"

      maxMonth = !maxMonth || ym > maxMonth ? ym : maxMonth;

      rows.push({
        ym,
        lsoa: (r.lsoa_code || "").trim(),
        burg: Number(r.burglary_count ?? r.burglary ?? 0), // ← burglary total in that row
      });
    });

    if (!rows.length)
      return showError("CSV parsed but no usable rows were found.");

    aggregate(rows, maxMonth);
  } catch (err) {
    showError("Crime CSV load failed: " + err.message);
  }
}

/* ---------- 3. aggregate ---------- */
function aggregate(rows, maxM) {
  /* 1. define the 12-month window (skip the most-recent 3 months) */
  const end = minusMonths(maxM, 3); // e.g. "2025-02"
  const start = minusMonths(end, 11); // 11 months earlier
  const months = [...monthRange(start, end)]; // chronological array

  monthsArr = months; // expose to the charts

  /* 2. start fresh each time */
  wardMonth.clear();
  lsoaCnt.clear();
  londonMean = {};

  /* 3. tally counts */
  rows.forEach(({ ym, lsoa, burg }) => {
    if (!months.includes(ym)) return; // outside our 12-month range
    const ward = lsoaToWard[lsoa];
    if (!ward) return; // unknown LSOA

    const wmKey = `${ym}|${ward}`;
    wardMonth.set(wmKey, (wardMonth.get(wmKey) || 0) + burg); // use burglary count

    const lwKey = `${ward}|${lsoa}`;
    lsoaCnt.set(lwKey, (lsoaCnt.get(lwKey) || 0) + burg);
  });

  /* 4. compute the London mean for each month */
  months.forEach((m) => {
    let total = 0,
      wardsWithData = 0;

    wardName.forEach((_, w) => {
      const v = wardMonth.get(`${m}|${w}`) || 0;
      if (v > 0) {
        total += v;
        wardsWithData++;
      }
    });

    londonMean[m] = wardsWithData ? total / wardsWithData : 0;
  });

  /* 5. (re)draw the UI */
  buildUI();
}

/* ---------- 4. UI + Plotly ---------- */
function buildUI() {
  const sel = $("wardSel");
  wardName.forEach((name, code) => {
    const opt = document.createElement("option");
    opt.value = code;
    opt.textContent = `${name} - ${code}`;
    sel.appendChild(opt);
  });
  $("status").style.display = "none";
  clearError();
  sel.onchange = () => render(sel.value);
  render(sel.value);
}

// ── rebuild the two plots with auto-sized Y-axes ──
function render(ward) {
  const wardSeries = monthsArr.map((m) => wardMonth.get(`${m}|${ward}`) || 0);
  const londonSer = monthsArr.map((m) => londonMean[m] || 0);

  /* ---------- line chart ---------- */
  const maxLine = Math.max(...wardSeries, ...londonSer, 1); // avoid 0-height axis
  Plotly.newPlot(
    "linePlot",
    [
      {
        x: monthsArr,
        y: londonSer,
        name: "London mean / ward",
        mode: "lines+markers",
        line: { dash: "dash" },
      },
      {
        x: monthsArr,
        y: wardSeries,
        name: `${ward} total`,
        mode: "lines+markers",
      },
    ],
    {
      yaxis: {
        title: "Burglaries",
        range: [0, Math.ceil(maxLine * 1.1)],
        fixedrange: true,
        rangemode: "tozero",
      },
      xaxis: { title: "Month", tickangle: -45, fixedrange: true },
      margin: { l: 40, r: 20, t: 40, b: 80 },
      legend: { orientation: "h", y: 1.15, x: 1 },
    },
    { responsive: true }
  );

  /* ---------- bar chart ---------- */
  const arr = [];
  lsoaCnt.forEach((cnt, key) => {
    const [w, l] = key.split("|");
    if (w === ward) arr.push([l, cnt]);
  });
  arr.sort((a, b) => b[1] - a[1]);

  const maxBar = Math.max(...arr.map((d) => d[1]), 1);
  Plotly.newPlot(
    "barPlot",
    [{ x: arr.map((d) => d[0]), y: arr.map((d) => d[1]), type: "bar" }],
    {
      yaxis: {
        title: "Burglaries",
        range: [0, Math.ceil(maxBar * 1.1)],
        fixedrange: true,
        rangemode: "tozero",
      },
      xaxis: { title: "LSOA", tickangle: 45, fixedrange: true },
      margin: { l: 40, r: 20, t: 40, b: 120 },
    },
    { responsive: true }
  );
}
