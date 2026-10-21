/* =============================================================================
 * app.js — CRISP-DM 多元線性回歸互動分析平台 (GitHub Pages 靜態網頁)
 *  - 下拉選單切換兩種 Kaggle Problem：King County 房價 / Flood Prediction
 *  - 可任意輸入 CSV Dataset URL (或上傳本機 CSV) 進行 MLR 分析
 *  - 特徵選擇：Forward Stepwise / Backward Elimination / SelectKBest / 全部 / 手動
 *  - 評估：R²、Adj R²、RMSE、MAE、MAPE、5-Fold CV、信賴區間 (CI) 與預測區間 (PI)
 * ===========================================================================*/
(() => {
  'use strict';

  // ------------------------------------------------------------------ 問題設定
  const PROBLEMS = {
    kc: {
      label: '🏠 Problem 1：King County 房價預測 (House Sales, 18 features)',
      url: 'https://raw.githubusercontent.com/Richard5007/AIoT_L3_CWA_HW2/main/kc_house_data.csv',
      fallbackUrl: 'kc_house_data.csv',
      target: 'price',
      logDefault: false,
      source: 'https://www.kaggle.com/datasets/harlfoxem/housesalesprediction',
      sourceName: 'Kaggle — House Sales in King County, USA (harlfoxem)',
      prepare: prepareKC,
      business: `
        <p><b>背景：</b>美國華盛頓州金郡 (含西雅圖、Bellevue、Redmond 等科技重鎮) 房市熱絡但高度非均質。
        銀行房貸審查、稅務估價與房仲需要一套<b>客觀、可解釋</b>的公允估價模型。</p>
        <p><b>業務問題：</b>在控制其他條件下，室內坪數、建材等級、景觀、屋齡、距市中心距離等因素，各自對房價帶來多少邊際溢價？</p>
        <p><b>ML 任務：</b>以多元線性回歸 (MLR) 預測連續目標 <code>price</code>，並提供點估計與信賴區間 / 預測區間。</p>
        <p><b>成功準則：</b>測試集 R² ≥ 0.70；特徵 VIF &lt; 10；殘差無嚴重系統性偏誤；提供 CI / PI。</p>`,
      benchmark: `
        <table class="tbl small"><thead><tr><th>參考方案 (本專案 Python 版 / 網路主流)</th><th>Test R²</th><th>RMSE</th></tr></thead><tbody>
        <tr><td>簡單線性回歸 (sqft_living)</td><td>0.494</td><td>$275,655</td></tr>
        <tr><td>精選特徵 MLR (Python 版 16 特徵)</td><td>0.716</td><td>$206,382</td></tr>
        <tr><td>Random Forest (非線性對照)</td><td>0.867</td><td>$141,395</td></tr>
        <tr><td>Kaggle 主流 GBDT (LightGBM / XGBoost + zipcode encoding)</td><td>0.87 ~ 0.90</td><td>$115k ~ $130k</td></tr>
        </tbody></table>
        <p class="muted">此資料集為 Kaggle Dataset (非競賽)，無官方排行榜；以上為公開 Notebook 常見成績區間。</p>`,
    },
    flood: {
      label: '🌊 Problem 2：Flood Prediction 洪水機率預測 (Playground S4E5, 20 features)',
      url: 'https://raw.githubusercontent.com/Richard5007/AIoT_L3_CWA_HW2/main/data/flood_sample.csv',
      fallbackUrl: 'data/flood_sample.csv',
      target: 'FloodProbability',
      logDefault: false,
      source: 'https://www.kaggle.com/competitions/playground-series-s4e5',
      sourceName: 'Kaggle — Regression with a Flood Prediction Dataset (Playground Series S4E5)',
      prepare: null,
      business: `
        <p><b>背景：</b>氣候變遷使極端降雨與洪災頻率上升。政府防災單位、保險公司需要量化各區域的洪水風險，
        以排定防洪預算、制定保費與預警機制。</p>
        <p><b>業務問題：</b>季風強度、地形排水、河川管理、森林砍伐、都市化、氣候變遷、水壩品質……等 20 項風險因子，
        各自對洪水發生機率的邊際影響為何？哪些因子最值得優先投資改善？</p>
        <p><b>ML 任務：</b>以多元線性回歸預測連續目標 <code>FloodProbability</code> (0~1)，並進行特徵選擇與區間估計。</p>
        <p><b>成功準則：</b>競賽評估指標為 <b>R²</b>；線性模型目標 R² ≥ 0.84，並能解釋每個因子的係數意義。</p>
        <p class="muted">資料說明：原始 train.csv 共 1,117,957 筆，為了在瀏覽器即時運算，本網頁使用隨機抽樣 20,000 筆 (seed=42) 的子集 <code>data/flood_sample.csv</code>。</p>`,
      benchmark: `
        <table class="tbl small"><thead><tr><th>Kaggle 排行榜參考 (Playground S4E5，評估指標 R²)</th><th>R²</th></tr></thead><tbody>
        <tr><td>Linear Regression / Ridge (原始 20 特徵)</td><td>≈ 0.845</td></tr>
        <tr><td>線性模型 + 列統計特徵 (row sum / std / max …)</td><td>≈ 0.865</td></tr>
        <tr><td>Top 名次方案 (CatBoost / LightGBM / NN ensemble)</td><td>≈ 0.869</td></tr>
        </tbody></table>
        <p class="muted">數值為公開討論區 / Notebook 之大約成績，實際名次請以 Kaggle Leaderboard 為準。</p>`,
    },
    custom: {
      label: '🔗 自訂 Dataset：任意輸入 CSV 網址 / 上傳檔案',
      url: null,
      target: null,
      logDefault: false,
      source: null,
      prepare: null,
      business: null,
      benchmark: null,
    },
  };

  const EXAMPLES = [
    { name: '50_Startups (Profit)', url: 'https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/50_Startups.csv' },
    { name: 'Boston Housing (medv)', url: 'https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv' },
    { name: 'Medical Insurance (charges)', url: 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv' },
  ];

  const PLOT_CFG = { responsive: true, displaylogo: false };
  const COLORS = { primary: '#4f46e5', teal: '#0d9488', amber: '#f59e0b', rose: '#e11d48', slate: '#64748b' };

  // ------------------------------------------------------------------ 狀態
  const S = { key: 'kc', raw: null, rawName: '', data: null, result: null };

  // ------------------------------------------------------------------ DOM 工具
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

  function fmt(v, d) {
    if (v === null || v === undefined || Number.isNaN(v)) return '—';
    if (!isFinite(v)) return v > 0 ? '∞' : '-∞';
    if (d !== undefined) return Number(v).toLocaleString('en-US', { minimumFractionDigits: d, maximumFractionDigits: d });
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e5) return Math.round(v).toLocaleString('en-US');
    if (a >= 100) return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
    if (a >= 0.01) return v.toLocaleString('en-US', { maximumFractionDigits: 4 });
    return v.toExponential(3);
  }
  const fmtP = (p) => (p < 1e-4 ? '&lt; 0.0001' : fmt(p, 4));

  function setStatus(msg, type = 'info') {
    const el = $('status');
    el.className = 'status ' + type;
    el.innerHTML = msg;
  }

  // ------------------------------------------------------------------ 資料載入
  function normalizeUrl(u) {
    u = u.trim();
    // GitHub blob 網址自動轉 raw
    const m = u.match(/^https?:\/\/github\.com\/([^/]+)\/([^/]+)\/blob\/(.+)$/);
    if (m) return `https://raw.githubusercontent.com/${m[1]}/${m[2]}/${m[3]}`;
    return u;
  }

  function parseCSV(input, isFile) {
    return new Promise((resolve, reject) => {
      Papa.parse(input, {
        download: !isFile,
        header: true,
        skipEmptyLines: true,
        dynamicTyping: false,
        complete: (res) => {
          if (!res.data || !res.data.length) return reject(new Error('CSV 無資料或格式錯誤'));
          const fields = res.meta.fields.filter(f => f !== undefined && f !== null && String(f).trim() !== '');
          resolve({ fields, rows: res.data });
        },
        error: (err) => {
          const msg = (err && (err.message || (err.type === 'error' ? '網路請求失敗 (請確認檔案存在或伺服器正常)' : JSON.stringify(err)))) || '載入失敗';
          reject(new Error(msg));
        },
      });
    });
  }

  async function loadProblem(key) {
    S.key = key;
    const P = PROBLEMS[key];
    $('customPanel').classList.toggle('hidden', key !== 'custom');
    $('targetRow').classList.toggle('hidden', key !== 'custom');
    if (key === 'custom') {
      setStatus('請輸入 CSV 網址 (例如 50_Startups.csv) 後按「載入資料」，或上傳本機 CSV 檔。', 'info');
      return;
    }
    $('logY').checked = P.logDefault;
    setStatus(`⏳ 正在載入資料集 …`, 'info');
    try {
      let raw;
      try {
        raw = await parseCSV(P.url, false);
      } catch (errOnline) {
        if (P.fallbackUrl) {
          raw = await parseCSV(P.fallbackUrl, false);
        } else {
          throw errOnline;
        }
      }
      onRawLoaded(raw, P.sourceName || P.url);
    } catch (e) {
      setStatus(`❌ 載入失敗：${esc(e.message || e)}（若在本地端直接以 file:// 協議開啟，瀏覽器會阻擋讀取本機 CSV，請使用本機伺服器如 <code>python -m http.server</code>，或直接點選上方的「或上傳 CSV」直接讀取本機檔案）`, 'error');
    }
  }

  async function loadCustomUrl() {
    const url = normalizeUrl($('urlInput').value);
    if (!url) return;
    $('urlInput').value = url;
    setStatus(`⏳ 正在下載 <b>${esc(url)}</b> …`, 'info');
    try {
      const raw = await parseCSV(url, false);
      onRawLoaded(raw, url);
    } catch (e) {
      setStatus(`❌ 無法讀取此網址：${esc(e.message || e)}<br>請確認是可公開存取、支援 CORS 的 <b>raw CSV</b> 連結 (例如 raw.githubusercontent.com)。Kaggle 下載頁需登入，請先下載後用「上傳 CSV」。`, 'error');
    }
  }

  async function loadCustomFile(file) {
    setStatus(`⏳ 正在讀取 <b>${esc(file.name)}</b> …`, 'info');
    try {
      const raw = await parseCSV(file, true);
      onRawLoaded(raw, file.name);
    } catch (e) {
      setStatus(`❌ 檔案解析失敗：${esc(e.message || e)}`, 'error');
    }
  }

  function onRawLoaded(raw, name) {
    S.raw = raw; S.rawName = name;
    const P = PROBLEMS[S.key];
    // 目標欄位下拉 (自訂資料集才顯示)
    const numericCols = raw.fields.filter(f => colInfo(raw, f).numeric);
    if (S.key === 'custom') {
      const sel = $('targetSelect');
      sel.innerHTML = numericCols.map(f => `<option value="${esc(f)}">${esc(f)}</option>`).join('');
      sel.value = numericCols[numericCols.length - 1];
    }
    const target = S.key === 'custom' ? $('targetSelect').value : P.target;
    if (!target) { setStatus('❌ 找不到任何數值欄位可當作目標變數 (target)。', 'error'); return; }
    prepareAndRun(target, true);
  }

  // ------------------------------------------------------------------ 資料準備
  const isMissing = (v) => v === undefined || v === null || String(v).trim() === '' || /^(na|nan|null|none|\?)$/i.test(String(v).trim());

  function colInfo(raw, f) {
    let nNum = 0, nMiss = 0;
    const uniq = new Set();
    for (const r of raw.rows) {
      const v = r[f];
      if (isMissing(v)) { nMiss++; continue; }
      if (uniq.size < 1000) uniq.add(String(v).trim());
      if (isFinite(Number(String(v).trim()))) nNum++;
    }
    const nonMiss = raw.rows.length - nMiss;
    return { numeric: nonMiss > 0 && nNum / nonMiss >= 0.97, nMiss, nUnique: uniq.size, levels: uniq };
  }

  /** King County 專屬特徵工程 (與 Python 主程式 7115056152_hw2.py 一致) */
  function prepareKC(raw, notes) {
    const LAT0 = 47.6062, LON0 = -122.3321;
    const before = raw.rows.length;
    const rows = raw.rows.filter(r => Number(r.bedrooms) < 30);
    notes.push(`移除筆誤離群值 <code>bedrooms ≥ 30</code> (33 房僅 1,620 sqft)：${before - rows.length} 筆。`);
    const out = rows.map(r => {
      const saleYear = Number(String(r.date).slice(0, 4));
      return {
        price: Number(r.price),
        bedrooms: +r.bedrooms, bathrooms: +r.bathrooms, sqft_living: +r.sqft_living, sqft_lot: +r.sqft_lot,
        floors: +r.floors, waterfront: +r.waterfront, view: +r.view, condition: +r.condition, grade: +r.grade,
        sqft_above: +r.sqft_above, sqft_basement: +r.sqft_basement,
        dist_to_seattle: Math.sqrt((r.lat - LAT0) ** 2 + (r.long - LON0) ** 2),
        lat: +r.lat, long: +r.long, sqft_living15: +r.sqft_living15, sqft_lot15: +r.sqft_lot15,
        house_age: saleYear - Number(r.yr_built), is_renovated: Number(r.yr_renovated) > 0 ? 1 : 0,
      };
    });
    notes.push('特徵工程：<code>house_age = 銷售年 − yr_built</code>、<code>is_renovated = (yr_renovated &gt; 0)</code>、<code>dist_to_seattle</code> = 與西雅圖市中心 (47.6062, −122.3321) 之距離。');
    notes.push('移除非預測欄位：<code>id</code>、<code>date</code>、<code>zipcode</code> (類別代碼)、<code>yr_built</code> / <code>yr_renovated</code> (已轉為衍生特徵)。');
    notes.push('⚠️ 已知完全共線性：<code>sqft_living = sqft_above + sqft_basement</code>，模型會自動偵測並剔除 aliased 欄位。');
    const fields = Object.keys(out[0]);
    return { fields, rows: out };
  }

  function prepareData(target) {
    const P = PROBLEMS[S.key];
    const notes = [];
    let raw = S.raw;
    if (P.prepare) raw = P.prepare(raw, notes);

    const ID_RE = /^(id|index|unnamed.*|row.?id|.*_id)$/i;
    const features = [], groups = [], dropped = [], colTable = [];
    // 目標
    let rows = raw.rows.filter(r => !isMissing(r[target]) && isFinite(Number(r[target])));
    if (rows.length < raw.rows.length) notes.push(`移除目標變數缺失 / 非數值之列：${raw.rows.length - rows.length} 筆。`);
    const y = Float64Array.from(rows, r => Number(r[target]));

    const cols = []; // {name, values: Float64Array}
    for (const f of raw.fields) {
      if (f === target) continue;
      const info = colInfo({ rows }, f);
      const base = { name: f, missing: info.nMiss, unique: info.nUnique };
      if (ID_RE.test(f) && info.nUnique > 0.9 * rows.length) {
        dropped.push(`${f} (識別碼)`); colTable.push({ ...base, type: '識別碼', role: '剔除' }); continue;
      }
      if (info.nUnique <= 1) { dropped.push(`${f} (常數)`); colTable.push({ ...base, type: '常數', role: '剔除' }); continue; }
      if (info.numeric) {
        const vals = rows.map(r => (isMissing(r[f]) ? NaN : Number(r[f])));
        const ok = vals.filter(v => !Number.isNaN(v));
        const med = Stats.median(ok);
        if (info.nMiss > 0) notes.push(`<code>${esc(f)}</code> 有 ${info.nMiss} 個缺失值，以中位數 ${fmt(med)} 補值。`);
        cols.push({ name: f, values: Float64Array.from(vals, v => (Number.isNaN(v) ? med : v)) });
        features.push(f);
        groups.push({ name: f, type: 'num', cols: [f] });
        colTable.push({ ...base, type: '數值', role: '候選特徵', mean: Stats.mean(ok), min: Math.min(...ok), max: Math.max(...ok) });
      } else if (info.nUnique <= 15) {
        // One-Hot Encoding (drop first：以最常見類別為基準)
        const cnt = {};
        rows.forEach(r => { const v = isMissing(r[f]) ? 'NA' : String(r[f]).trim(); cnt[v] = (cnt[v] || 0) + 1; });
        const levels = Object.keys(cnt).sort((a, b) => cnt[b] - cnt[a]);
        const baseLv = levels[0];
        const gcols = [];
        for (const lv of levels.slice(1)) {
          const name = `${f}_${lv}`;
          cols.push({ name, values: Float64Array.from(rows, r => ((isMissing(r[f]) ? 'NA' : String(r[f]).trim()) === lv ? 1 : 0)) });
          features.push(name); gcols.push(name);
        }
        groups.push({ name: f, type: 'cat', levels, base: baseLv, cols: gcols });
        notes.push(`類別欄位 <code>${esc(f)}</code> 進行 One-Hot Encoding (${levels.length} 類，基準類別 = <b>${esc(baseLv)}</b>，避免虛擬變數陷阱)。`);
        colTable.push({ ...base, type: '類別', role: `One-Hot (${levels.length - 1} 欄)` });
      } else {
        dropped.push(`${f} (高基數文字)`);
        colTable.push({ ...base, type: '文字', role: '剔除' });
      }
    }
    if (dropped.length) notes.push(`剔除欄位：${dropped.map(esc).join('、')}。`);
    const n = rows.length, p = features.length;
    const X = Array.from({ length: n }, (_, i) => {
      const r = new Float64Array(p);
      for (let j = 0; j < p; j++) r[j] = cols[j].values[i];
      return r;
    });
    return { target, features, groups, X, y, notes, colTable, nRaw: S.raw.rows.length, rawCols: S.raw.fields.length };
  }

  function prepareAndRun(target, resetPool) {
    try {
      S.data = prepareData(target);
    } catch (e) {
      console.error(e);
      setStatus(`❌ 資料準備失敗：${esc(e.message)}`, 'error');
      return;
    }
    const D = S.data;
    if (D.features.length < 1) { setStatus('❌ 沒有可用的特徵欄位。', 'error'); return; }
    if (resetPool) buildPool(D);
    const allPos = D.y.every(v => v > 0);
    $('logY').disabled = !allPos;
    if (!allPos) $('logY').checked = false;
    $('kSlider').max = D.features.length;
    if (+$('kSlider').value > D.features.length || resetPool) $('kSlider').value = Math.min(D.features.length, Math.max(1, Math.round(D.features.length * 0.6)));
    $('kVal').textContent = $('kSlider').value;
    run();
  }

  function buildPool(D) {
    $('poolBox').innerHTML = D.features.map((f, j) =>
      `<label class="chk"><input type="checkbox" data-j="${j}" checked> ${esc(f)}</label>`).join('');
    $('poolCount').textContent = `${D.features.length} / ${D.features.length}`;
  }

  function getPool() {
    const boxes = [...$('poolBox').querySelectorAll('input[type=checkbox]')];
    return boxes.filter(b => b.checked).map(b => +b.dataset.j);
  }

  // ------------------------------------------------------------------ 特徵選擇
  function critValue(fit, crit) {
    if (crit === 'aic') return fit.aic;
    if (crit === 'bic') return fit.bic;
    return -fit.adjR2;
  }

  function forwardSelect(K, pool, crit) {
    const selected = [], path = [];
    let remaining = pool.slice();
    while (remaining.length) {
      let best = null;
      for (const j of remaining) {
        const f = Stats.fitSubset(K, [...selected, j]);
        if (f.aliased.length) continue;
        const sc = critValue(f, crit);
        if (!best || sc < best.sc) best = { j, f, sc };
      }
      if (!best) break;
      selected.push(best.j);
      remaining = remaining.filter(j => j !== best.j);
      path.push({ k: selected.length, add: best.j, feats: selected.slice(), fit: best.f, sc: best.sc });
    }
    let bi = 0;
    path.forEach((s, i) => { if (s.sc < path[bi].sc) bi = i; });
    return { path, feats: path.length ? path[bi].feats : [], bestIdx: bi };
  }

  function coefPValues(fit) {
    return fit.b.map((b, j) => {
      const se = Math.sqrt(fit.sigma2 * fit.Cinv[j][j]);
      return Stats.tPValue(b / se, fit.df);
    });
  }

  function backwardSelect(K, pool, alpha) {
    let S0 = Stats.nonAliased(K.C, pool);
    const steps = [];
    while (S0.length) {
      const f = Stats.fitSubset(K, S0);
      const ps = coefPValues(f);
      let wi = 0;
      ps.forEach((p, i) => { if (p > ps[wi]) wi = i; });
      if (ps[wi] <= alpha) break;
      steps.push({ removed: f.kept[wi], p: ps[wi] });
      S0 = f.kept.filter((_, i) => i !== wi);
    }
    return { feats: S0, steps };
  }

  function kBestSelect(D, train, pool, k) {
    const yTr = train.map(i => D.y[i]);
    const scores = pool.map(j => ({ j, r: Stats.corr(train.map(i => D.X[i][j]), yTr) }));
    scores.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    return { feats: scores.slice(0, k).map(s => s.j), scores };
  }

  // ------------------------------------------------------------------ 主分析流程
  function readSettings() {
    return {
      method: $('methodSelect').value,
      crit: $('critSelect').value,
      k: +$('kSlider').value,
      alpha: +$('alphaSelect').value,
      testRatio: +$('testSlider').value / 100,
      conf: +$('confSelect').value,
      logY: $('logY').checked && !$('logY').disabled,
      seed: +$('seedInput').value || 42,
    };
  }

  function run() {
    if (!S.data) return;
    const t0 = performance.now();
    try {
      S.result = analyze(S.data, readSettings());
      render(S.data, S.result);
      const dt = (performance.now() - t0).toFixed(0);
      const R = S.result;
      setStatus(`✅ 已完成分析：<b>${esc(S.rawName)}</b> ｜ ${R.n.toLocaleString()} 筆、${S.data.features.length} 個候選特徵 → 選出 <b>${R.fit.k}</b> 個特徵 ｜ Test R² = <b>${fmt(R.testM.r2, 4)}</b> ｜ 耗時 ${dt} ms`, 'ok');
    } catch (e) {
      console.error(e);
      setStatus(`❌ 分析失敗：${esc(e.message)}`, 'error');
    }
  }

  function analyze(D, cfg) {
    const n = D.y.length, p = D.features.length;
    const yModel = cfg.logY ? Float64Array.from(D.y, Math.log) : D.y;
    // 1) Train / Test split
    const idx = Stats.shuffle(n, cfg.seed);
    const nTest = Math.max(1, Math.round(n * cfg.testRatio));
    const test = idx.slice(0, nTest), train = idx.slice(nTest);
    if (train.length < 3) throw new Error('訓練資料太少');
    // 2) 以訓練集平均 / 標準差標準化 (避免資料洩漏)
    const mu = new Float64Array(p), sd = new Float64Array(p);
    for (const i of train) for (let j = 0; j < p; j++) mu[j] += D.X[i][j];
    for (let j = 0; j < p; j++) mu[j] /= train.length;
    for (const i of train) for (let j = 0; j < p; j++) sd[j] += (D.X[i][j] - mu[j]) ** 2;
    for (let j = 0; j < p; j++) sd[j] = Math.sqrt(sd[j] / train.length) || 1;
    const Z = D.X.map(r => { const z = new Float64Array(p); for (let j = 0; j < p; j++) z[j] = (r[j] - mu[j]) / sd[j]; return z; });
    const constCols = [];
    for (let j = 0; j < p; j++) {
      let mn = Infinity, mx = -Infinity;
      for (const i of train) { const v = D.X[i][j]; if (v < mn) mn = v; if (v > mx) mx = v; }
      if (mn === mx) constCols.push(j);
    }
    // 3) 充分統計量
    const Gtr = Stats.gram(Z, yModel, train);
    const K = Stats.center(Gtr);
    let pool = getPool().filter(j => !constCols.includes(j));
    if (!pool.length) throw new Error('候選特徵池為空，請至少勾選一個特徵');

    // 4) 各種特徵選擇方法
    const fwd = forwardSelect(K, pool, cfg.crit);
    const bwd = backwardSelect(K, pool, cfg.alpha);
    const kb = kBestSelect(D, train, pool, Math.min(cfg.k, pool.length));
    const methods = {
      all: { name: '全部候選特徵 (Full MLR)', feats: pool },
      forward: { name: `Forward Stepwise (${cfg.crit.toUpperCase()})`, feats: fwd.feats },
      backward: { name: `Backward Elimination (p > ${cfg.alpha})`, feats: bwd.feats },
      kbest: { name: `SelectKBest |r| (k=${Math.min(cfg.k, pool.length)})`, feats: kb.feats },
      manual: { name: '手動勾選特徵', feats: pool },
    };
    // 5) K-Fold CV 統計量 (5 folds，於訓練集內)
    const folds = 5, foldRows = Array.from({ length: folds }, () => []);
    train.forEach((i, t) => foldRows[t % folds].push(i));
    const foldG = foldRows.map(rows => Stats.gram(Z, yModel, rows));

    const back = (v) => (cfg.logY ? Math.exp(v) : v);
    const tCrit = (df) => Stats.tInv(1 - (1 - cfg.conf) / 2, df);

    function evaluate(feats) {
      const fit = Stats.fitSubset(K, feats);
      const yhatTe = test.map(i => back(Stats.predictZ(fit, Z[i]).yhat));
      const yTe = test.map(i => D.y[i]);
      const m = Stats.metrics(yTe, yhatTe, fit.k);
      // CV
      let cvSse = 0, cvN = 0;
      for (let f = 0; f < folds; f++) {
        const Kf = Stats.center(Stats.subtractGram(Gtr, foldG[f]));
        const ff = Stats.fitSubset(Kf, fit.kept);
        for (const i of foldRows[f]) { const e = D.y[i] - back(Stats.predictZ(ff, Z[i]).yhat); cvSse += e * e; cvN++; }
      }
      return { fit, testM: m, cvRmse: Math.sqrt(cvSse / cvN) };
    }

    const comparison = {};
    for (const key of ['all', 'forward', 'backward', 'kbest']) comparison[key] = { ...methods[key], ...evaluate(methods[key].feats) };
    const chosenKey = cfg.method;
    const chosen = comparison[chosenKey] || { ...methods[chosenKey], ...evaluate(methods[chosenKey].feats) };
    const fit = chosen.fit;

    // forward path 的 test R²
    const pathEval = fwd.path.map(s => {
      const yhat = test.map(i => back(Stats.predictZ(s.fit, Z[i]).yhat));
      return { k: s.k, add: D.features[s.add], trainAdjR2: s.fit.adjR2, testR2: Stats.metrics(test.map(i => D.y[i]), yhat, s.k).r2, sc: s.sc };
    });

    // 6) 係數表 (原始尺度)
    const tc = tCrit(fit.df);
    const sdY = Math.sqrt(K.syy / K.n);
    const coefs = fit.kept.map((j, a) => {
      const seZ = Math.sqrt(fit.sigma2 * fit.Cinv[a][a]);
      const beta = fit.b[a] / sd[j], se = seZ / sd[j], t = fit.b[a] / seZ;
      return {
        j, name: D.features[j], beta, se, t, p: Stats.tPValue(t, fit.df),
        lo: beta - tc * se, hi: beta + tc * se,
        std: fit.b[a] / sdY, stdLo: (fit.b[a] - tc * seZ) / sdY, stdHi: (fit.b[a] + tc * seZ) / sdY,
        vif: K.C[j][j] * fit.Cinv[a][a],
      };
    });
    const z0 = Array.from(mu, (m, j) => -m / sd[j]);
    const ip = Stats.predictZ(fit, z0);
    const seI = Math.sqrt(fit.sigma2 * ip.lev);
    const intercept = { name: '(Intercept)', beta: ip.yhat, se: seI, t: ip.yhat / seI, p: Stats.tPValue(ip.yhat / seI, fit.df), lo: ip.yhat - tc * seI, hi: ip.yhat + tc * seI };
    const ssr = K.syy - fit.sse;
    const F = fit.k ? (ssr / fit.k) / fit.sigma2 : NaN;
    const Fp = fit.k ? Stats.fPValue(F, fit.k, fit.df) : NaN;

    // 7) 預測與區間 (測試集)
    const preds = test.map(i => {
      const { yhat, lev } = Stats.predictZ(fit, Z[i]);
      const seF = Math.sqrt(fit.sigma2 * lev), seP = Math.sqrt(fit.sigma2 * (1 + lev));
      return {
        i, y: D.y[i], yhat: back(yhat), yhatM: yhat, yM: yModel[i],
        ciLo: back(yhat - tc * seF), ciHi: back(yhat + tc * seF),
        piLo: back(yhat - tc * seP), piHi: back(yhat + tc * seP),
      };
    });
    const coverage = preds.filter(q => q.y >= q.piLo && q.y <= q.piHi).length / preds.length;
    const trainPred = train.map(i => back(Stats.predictZ(fit, Z[i]).yhat));
    const trainM = Stats.metrics(train.map(i => D.y[i]), trainPred, fit.k);
    const residTr = train.map((i) => yModel[i] - Stats.predictZ(fit, Z[i]).yhat);
    const fittedTr = train.map((i) => Stats.predictZ(fit, Z[i]).yhat);

    // 8) 簡單線性回歸 (最相關單一特徵) + CI/PI
    const topJ = coefs.length ? coefs.slice().sort((a, b) => Math.abs(b.std) - Math.abs(a.std))[0].j
      : pool[0];
    const slr = Stats.fitSubset(K, [topJ]);
    const tcS = tCrit(slr.df);

    return {
      cfg, n, train, test, mu, sd, Z, yModel, K, pool, constCols, fwd, bwd, kb, pathEval, methods, comparison, chosenKey, chosen,
      fit, coefs, intercept, F, Fp, preds, coverage, trainM, testM: chosen.testM, residTr, fittedTr, slr, topJ, tcS, tc, back,
    };
  }

  // ------------------------------------------------------------------ 渲染
  function render(D, R) {
    renderBusiness(D, R);
    renderDataUnderstanding(D, R);
    renderPreparation(D, R);
    renderModeling(D, R);
    renderEvaluation(D, R);
    renderDeployment(D, R);
  }

  function renderBusiness(D) {
    const P = PROBLEMS[S.key];
    const src = P.source
      ? `<p><b>資料集來源：</b><a href="${P.source}" target="_blank" rel="noopener">${esc(P.sourceName)}</a></p>`
      : `<p><b>資料集來源：</b><a href="${esc(S.rawName)}" target="_blank" rel="noopener">${esc(S.rawName)}</a> (使用者自訂)</p>`;
    const biz = P.business || `
      <p><b>業務問題：</b>以資料集中的其他欄位，預測連續型目標變數 <code>${esc(D.target)}</code>，並量化每個因子對目標的邊際影響。</p>
      <p><b>ML 任務：</b>多元線性回歸 (MLR) + 特徵選擇 + 模型評估 + 信賴 / 預測區間。</p>
      <p><b>成功準則：</b>測試集 R² 越高越好、VIF &lt; 10、殘差近似常態且無明顯異質變異。</p>`;
    $('bizContent').innerHTML = src + biz;
  }

  function renderDataUnderstanding(D, R) {
    const y = D.y;
    $('duKpi').innerHTML = kpis([
      ['原始筆數', D.nRaw.toLocaleString()], ['原始欄位', D.rawCols], ['建模筆數', y.length.toLocaleString()],
      ['候選特徵', D.features.length], ['目標平均', fmt(Stats.mean(y))], ['目標偏度', fmt(Stats.skew(Array.from(y)), 3)],
    ]);
    $('colTable').innerHTML = `<table class="tbl small"><thead><tr><th>欄位</th><th>型態</th><th>缺失</th><th>唯一值</th><th>平均</th><th>最小</th><th>最大</th><th>處理</th></tr></thead><tbody>
      <tr class="hl"><td><b>${esc(D.target)}</b></td><td>數值</td><td>0</td><td>—</td><td>${fmt(Stats.mean(y))}</td><td>${fmt(Math.min(...y))}</td><td>${fmt(Math.max(...y))}</td><td>🎯 目標變數</td></tr>
      ${D.colTable.map(c => `<tr><td>${esc(c.name)}</td><td>${c.type}</td><td>${c.missing}</td><td>${c.unique >= 1000 ? '≥1000' : c.unique}</td><td>${fmt(c.mean)}</td><td>${fmt(c.min)}</td><td>${fmt(c.max)}</td><td>${esc(c.role)}</td></tr>`).join('')}
      </tbody></table>`;

    // 目標分佈
    const traces = [{ x: Array.from(y), type: 'histogram', nbinsx: 50, marker: { color: COLORS.primary, opacity: 0.8 }, name: D.target }];
    const layout = baseLayout(`目標變數 ${D.target} 分佈 (Skewness = ${fmt(Stats.skew(Array.from(y)), 2)})`, D.target, '次數');
    if (y.every(v => v > 0)) {
      traces.push({ x: Array.from(y, Math.log), type: 'histogram', nbinsx: 50, marker: { color: COLORS.teal, opacity: 0.8 }, name: `log(${D.target})`, xaxis: 'x2', yaxis: 'y2' });
      Object.assign(layout, { grid: { rows: 1, columns: 2, pattern: 'independent' }, xaxis2: { title: `log(${D.target})` }, showlegend: false });
    }
    Plotly.react('chartTarget', traces, layout, PLOT_CFG);

    // 與目標相關係數
    const corrs = D.features.map((f, j) => ({ f, j, r: Stats.corr(D.X.map(r => r[j]), y) }));
    corrs.sort((a, b) => Math.abs(b.r) - Math.abs(a.r));
    const top = corrs.slice(0, 25).reverse();
    Plotly.react('chartCorrBar', [{
      type: 'bar', orientation: 'h', y: top.map(c => c.f), x: top.map(c => c.r),
      marker: { color: top.map(c => (c.r >= 0 ? COLORS.primary : COLORS.rose)) },
      text: top.map(c => c.r.toFixed(3)), textposition: 'auto',
    }], { ...baseLayout(`各特徵與 ${D.target} 之 Pearson 相關係數`, 'r', ''), height: Math.max(360, top.length * 22 + 80), margin: { l: 170, r: 20, t: 50, b: 40 } }, PLOT_CFG);

    // 熱力圖
    const hm = corrs.slice(0, 14);
    const vars = [D.target, ...hm.map(c => c.f)];
    const series = [Array.from(y), ...hm.map(c => D.X.map(r => r[c.j]))];
    const M = series.map(a => series.map(b => Stats.corr(a, b)));
    Plotly.react('chartHeatmap', [{
      z: M, x: vars, y: vars, type: 'heatmap', colorscale: 'RdBu', reversescale: true, zmin: -1, zmax: 1,
      text: M.map(r => r.map(v => v.toFixed(2))), texttemplate: '%{text}', textfont: { size: 9 },
    }], { ...baseLayout('相關係數熱力圖 (目標 + 前 14 名特徵)', '', ''), height: 560, margin: { l: 140, r: 20, t: 50, b: 130 }, yaxis: { autorange: 'reversed' } }, PLOT_CFG);
  }

  function renderPreparation(D, R) {
    const cfg = R.cfg;
    const steps = [...D.notes];
    if (R.constCols.length) steps.push(`訓練集中為常數的欄位不納入：${R.constCols.map(j => esc(D.features[j])).join('、')}。`);
    steps.push(`資料切分：訓練集 <b>${R.train.length.toLocaleString()}</b> 筆 (${Math.round((1 - cfg.testRatio) * 100)}%)、測試集 <b>${R.test.length.toLocaleString()}</b> 筆 (${Math.round(cfg.testRatio * 100)}%)，random seed = ${cfg.seed}。`);
    steps.push('標準化：以<b>訓練集</b>平均與標準差做 Z-score (避免資料洩漏)；係數表會轉換回原始單位。');
    if (cfg.logY) steps.push(`目標變數取對數 <code>log(${esc(D.target)})</code> 以降低右偏；評估指標與區間皆轉換回原始尺度。`);
    $('prepList').innerHTML = steps.map(s => `<li>${s}</li>`).join('');
  }

  function renderModeling(D, R) {
    const F = (j) => esc(D.features[j]);
    // 方法說明
    const c = R.comparison;
    const bwdTxt = R.bwd.steps.length ? R.bwd.steps.map(s => `${F(s.removed)} (p=${fmt(s.p, 3)})`).join(' → ') : '無 (所有特徵皆顯著)';
    $('fsSummary').innerHTML = `
      <ul>
        <li><b>Forward Stepwise：</b>從空模型開始，每步加入使 ${R.cfg.crit.toUpperCase()} 最佳的特徵，最佳步數 = <b>${R.fwd.feats.length}</b>。</li>
        <li><b>Backward Elimination：</b>從全模型開始，逐步剔除 p-value 最大且 &gt; ${R.cfg.alpha} 的特徵。剔除順序：${bwdTxt}。</li>
        <li><b>SelectKBest：</b>依訓練集 |Pearson r| 排序取前 k = ${c.kbest.feats.length} 個。</li>
        <li><b>完全共線性偵測：</b>${c.all.fit.aliased.length ? `自動剔除 aliased 欄位 ${c.all.fit.aliased.map(F).join('、')}` : '未偵測到完全共線性'}。</li>
      </ul>`;

    // Forward path 圖
    const pe = R.pathEval;
    Plotly.react('chartPath', [
      { x: pe.map(s => s.k), y: pe.map(s => s.trainAdjR2), mode: 'lines+markers', name: 'Train Adj R²', line: { color: COLORS.primary }, text: pe.map(s => `+ ${s.add}`), hovertemplate: 'k=%{x}<br>%{text}<br>Adj R²=%{y:.4f}<extra></extra>' },
      { x: pe.map(s => s.k), y: pe.map(s => s.testR2), mode: 'lines+markers', name: 'Test R²', line: { color: COLORS.amber }, text: pe.map(s => `+ ${s.add}`), hovertemplate: 'k=%{x}<br>%{text}<br>Test R²=%{y:.4f}<extra></extra>' },
    ], {
      ...baseLayout('Forward Stepwise 路徑：特徵數 vs. R²', '特徵數 k', 'R²'),
      shapes: pe.length ? [{ type: 'line', x0: R.fwd.feats.length, x1: R.fwd.feats.length, yref: 'paper', y0: 0, y1: 1, line: { dash: 'dot', color: COLORS.rose } }] : [],
      annotations: pe.length ? [{ x: R.fwd.feats.length, yref: 'paper', y: 1.02, text: `最佳 k=${R.fwd.feats.length}`, showarrow: false, font: { color: COLORS.rose } }] : [],
    }, PLOT_CFG);

    // 方法比較表
    const rows = Object.entries(c).map(([key, m]) => `<tr class="${key === R.chosenKey ? 'hl' : ''}">
      <td>${key === R.chosenKey ? '★ ' : ''}${esc(m.name)}</td><td>${m.fit.k}</td><td>${fmt(m.fit.adjR2, 4)}</td>
      <td>${fmt(m.testM.r2, 4)}</td><td>${fmt(m.testM.rmse)}</td><td>${fmt(m.testM.mae)}</td><td>${fmt(m.cvRmse)}</td><td>${fmt(m.fit.aic, 1)}</td></tr>`).join('');
    const manualRow = R.chosenKey === 'manual' ? `<tr class="hl"><td>★ ${esc(R.chosen.name)}</td><td>${R.fit.k}</td><td>${fmt(R.fit.adjR2, 4)}</td><td>${fmt(R.chosen.testM.r2, 4)}</td><td>${fmt(R.chosen.testM.rmse)}</td><td>${fmt(R.chosen.testM.mae)}</td><td>${fmt(R.chosen.cvRmse)}</td><td>${fmt(R.fit.aic, 1)}</td></tr>` : '';
    $('methodTable').innerHTML = `<table class="tbl"><thead><tr><th>特徵選擇方法</th><th>k</th><th>Train Adj R²</th><th>Test R²</th><th>Test RMSE</th><th>Test MAE</th><th>5-Fold CV RMSE</th><th>AIC</th></tr></thead><tbody>${rows}${manualRow}</tbody></table>`;

    // 係數表
    const fit = R.fit, conf = Math.round(R.cfg.conf * 100);
    const coefRow = (c0, isInt) => `<tr><td>${esc(c0.name)}</td><td>${fmt(c0.beta)}</td><td>${fmt(c0.se)}</td><td>${fmt(c0.t, 3)}</td>
      <td class="${c0.p < 0.05 ? 'sig' : 'nsig'}">${fmtP(c0.p)}</td><td>[${fmt(c0.lo)}, ${fmt(c0.hi)}]</td>
      <td>${isInt ? '—' : fmt(c0.std, 4)}</td><td class="${!isInt && c0.vif > 10 ? 'nsig' : ''}">${isInt ? '—' : fmt(c0.vif, 2)}</td></tr>`;
    $('coefTable').innerHTML = `
      <div class="kpis">${kpis([
        ['選用方法', esc(R.chosen.name)], ['特徵數 k', fit.k], ['Train R²', fmt(fit.r2, 4)], ['Train Adj R²', fmt(fit.adjR2, 4)],
        ['F 統計量', fmt(R.F, 1)], ['F 檢定 p', fmtP(R.Fp)], ['殘差標準誤 σ', fmt(Math.sqrt(fit.sigma2))], ['自由度 df', fit.df.toLocaleString()],
      ], true)}</div>
      <table class="tbl small"><thead><tr><th>變數</th><th>係數 β</th><th>標準誤</th><th>t</th><th>p-value</th><th>${conf}% CI</th><th>標準化 β*</th><th>VIF</th></tr></thead>
      <tbody>${coefRow(R.intercept, true)}${R.coefs.map(c0 => coefRow(c0, false)).join('')}</tbody></table>
      <p class="muted">模型方程式 (${R.cfg.logY ? 'log(' + esc(D.target) + ')' : esc(D.target)})：ŷ = ${fmt(R.intercept.beta)} ${R.coefs.map(c0 => `${c0.beta >= 0 ? '+' : '−'} ${fmt(Math.abs(c0.beta))}·${esc(c0.name)}`).join(' ')}</p>`;

    const cs = R.coefs.slice().sort((a, b) => Math.abs(a.std) - Math.abs(b.std));
    Plotly.react('chartCoef', [{
      type: 'bar', orientation: 'h', y: cs.map(c0 => c0.name), x: cs.map(c0 => c0.std),
      error_x: { type: 'data', symmetric: false, array: cs.map(c0 => c0.stdHi - c0.std), arrayminus: cs.map(c0 => c0.std - c0.stdLo), color: '#334155' },
      marker: { color: cs.map(c0 => (c0.p < 0.05 ? (c0.std >= 0 ? COLORS.primary : COLORS.rose) : '#cbd5e1')) },
    }], { ...baseLayout(`標準化係數 β* 與 ${conf}% 信賴區間 (灰色 = 不顯著 p ≥ 0.05)`, 'β* (每 1 個標準差變動對 y 的標準差影響)', ''), height: Math.max(360, cs.length * 24 + 90), margin: { l: 170, r: 20, t: 50, b: 50 } }, PLOT_CFG);
  }

  function renderEvaluation(D, R) {
    const conf = Math.round(R.cfg.conf * 100);
    const tm = R.testM, trm = R.trainM;
    $('evalKpi').innerHTML = kpis([
      ['Test R²', fmt(tm.r2, 4)], ['Test Adj R²', fmt(tm.adjR2, 4)], ['Test RMSE', fmt(tm.rmse)], ['Test MAE', fmt(tm.mae)],
      ['Test MAPE', isNaN(tm.mape) ? '—' : fmt(tm.mape, 2) + '%'], ['5-Fold CV RMSE', fmt(R.chosen.cvRmse)],
      ['Train R²', fmt(trm.r2, 4)], [`${conf}% PI 涵蓋率`, fmt(R.coverage * 100, 1) + '%'],
    ]);
    const gap = trm.r2 - tm.r2;
    $('evalNote').innerHTML = `Train R² − Test R² = <b>${fmt(gap, 4)}</b> → ${Math.abs(gap) < 0.03 ? '✅ 無明顯過擬合' : gap > 0 ? '⚠️ 可能有過擬合' : 'ℹ️ 測試集表現略優 (抽樣變異)'}；
      ${conf}% 預測區間實際涵蓋 <b>${fmt(R.coverage * 100, 1)}%</b> 的測試樣本 ${Math.abs(R.coverage - R.cfg.conf) < 0.03 ? '(接近名目水準，區間校準良好)' : '(偏離名目水準，可能存在異質變異或非常態殘差)'}。`;
    const P = PROBLEMS[S.key];
    $('benchmark').innerHTML = P.benchmark
      ? `<h4>🏆 Kaggle 名次 / 外部基準比較</h4>${P.benchmark}<p>本網頁目前模型 Test R² = <b>${fmt(tm.r2, 4)}</b>，RMSE = <b>${fmt(tm.rmse)}</b>。</p>`
      : '';

    // 抽樣點，避免繪圖過慢
    const pts = R.preds.length > 2500 ? sample(R.preds, 2500, 7) : R.preds;
    const all = pts.flatMap(q => [q.y, q.yhat]);
    const lo = Math.min(...all), hi = Math.max(...all);
    Plotly.react('chartAvP', [
      {
        x: pts.map(q => q.y), y: pts.map(q => q.yhat), mode: 'markers', type: 'scattergl', name: '測試樣本',
        marker: { size: 5, color: pts.map(q => (q.y >= q.piLo && q.y <= q.piHi ? COLORS.primary : COLORS.rose)), opacity: 0.55 },
        error_y: pts.length <= 300 ? { type: 'data', symmetric: false, array: pts.map(q => q.piHi - q.yhat), arrayminus: pts.map(q => q.yhat - q.piLo), color: 'rgba(79,70,229,0.25)' } : undefined,
      },
      { x: [lo, hi], y: [lo, hi], mode: 'lines', name: 'y = x (完美預測)', line: { color: COLORS.amber, dash: 'dash' } },
    ], baseLayout(`實際值 vs. 預測值 (測試集；紅點 = 落在 ${conf}% PI 之外)`, `實際 ${D.target}`, `預測 ${D.target}`), PLOT_CFG);

    // 排序後預測區間圖
    const sub = sample(R.preds, Math.min(150, R.preds.length), 11).sort((a, b) => a.yhat - b.yhat);
    const xs = sub.map((_, i) => i + 1);
    Plotly.react('chartPI', [
      { x: xs, y: sub.map(q => q.piHi), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: xs, y: sub.map(q => q.piLo), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(13,148,136,0.18)', name: `${conf}% 預測區間 (PI)` },
      { x: xs, y: sub.map(q => q.ciHi), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: xs, y: sub.map(q => q.ciLo), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(79,70,229,0.45)', name: `${conf}% 信賴區間 (CI)` },
      { x: xs, y: sub.map(q => q.yhat), mode: 'lines', line: { color: COLORS.primary, width: 2 }, name: '預測值 ŷ' },
      { x: xs, y: sub.map(q => q.y), mode: 'markers', name: '實際值 y', marker: { size: 6, color: sub.map(q => (q.y >= q.piLo && q.y <= q.piHi ? COLORS.amber : COLORS.rose)), line: { width: 0.5, color: '#fff' } } },
    ], baseLayout(`測試集預測圖 + ${conf}% CI / PI (隨機 ${sub.length} 筆，依預測值排序)`, '樣本 (依 ŷ 排序)', D.target), PLOT_CFG);

    // 簡單線性回歸 + CI/PI
    renderSLR(D, R);

    // 殘差診斷
    const rs = R.residTr, fv = R.fittedTr;
    const ridx = rs.length > 3000 ? sample(rs.map((_, i) => i), 3000, 5) : rs.map((_, i) => i);
    const yLab = R.cfg.logY ? `log(${D.target})` : D.target;
    Plotly.react('chartResid', [
      { x: ridx.map(i => fv[i]), y: ridx.map(i => rs[i]), mode: 'markers', type: 'scattergl', marker: { size: 4, color: COLORS.primary, opacity: 0.45 }, name: '殘差' },
      { x: [Math.min(...fv), Math.max(...fv)], y: [0, 0], mode: 'lines', line: { color: COLORS.rose, dash: 'dash' }, name: '0' },
    ], { ...baseLayout('殘差 vs. 擬合值 (訓練集) — 檢查線性與同質變異', `擬合值 (${yLab})`, '殘差'), showlegend: false }, PLOT_CFG);

    const sorted = ridx.map(i => rs[i]).sort((a, b) => a - b);
    const sdR = Stats.std(sorted), mR = Stats.mean(sorted);
    const theo = sorted.map((_, i) => Stats.normInv((i + 0.5) / sorted.length));
    Plotly.react('chartQQ', [
      { x: theo, y: sorted.map(v => (v - mR) / sdR), mode: 'markers', type: 'scattergl', marker: { size: 4, color: COLORS.teal, opacity: 0.6 }, name: '標準化殘差' },
      { x: [theo[0], theo[theo.length - 1]], y: [theo[0], theo[theo.length - 1]], mode: 'lines', line: { color: COLORS.rose, dash: 'dash' }, name: '常態參考線' },
    ], { ...baseLayout(`Normal Q-Q Plot — 殘差常態性 (偏度 ${fmt(Stats.skew(rs), 2)})`, '理論分位數', '標準化殘差分位數'), showlegend: false }, PLOT_CFG);

    Plotly.react('chartResHist', [{ x: rs, type: 'histogram', nbinsx: 60, marker: { color: COLORS.slate } }],
      baseLayout('殘差分佈直方圖', '殘差', '次數'), PLOT_CFG);
  }

  function renderSLR(D, R) {
    const j = R.topJ, slr = R.slr, conf = Math.round(R.cfg.conf * 100);
    const fname = D.features[j];
    const pts = sample(R.test, Math.min(1500, R.test.length), 3);
    const xs = pts.map(i => D.X[i][j]), ys = pts.map(i => R.yModel[i]);
    const xmin = Math.min(...D.X.map(r => r[j])), xmax = Math.max(...D.X.map(r => r[j]));
    const grid = Array.from({ length: 80 }, (_, g) => xmin + (xmax - xmin) * g / 79);
    const z = new Float64Array(D.features.length);
    const band = grid.map(x => {
      z[j] = (x - R.mu[j]) / R.sd[j];
      const { yhat, lev } = Stats.predictZ(slr, z);
      const seF = Math.sqrt(slr.sigma2 * lev), seP = Math.sqrt(slr.sigma2 * (1 + lev));
      return { yhat, ciLo: yhat - R.tcS * seF, ciHi: yhat + R.tcS * seF, piLo: yhat - R.tcS * seP, piHi: yhat + R.tcS * seP };
    });
    const yLab = R.cfg.logY ? `log(${D.target})` : D.target;
    Plotly.react('chartSLR', [
      { x: xs, y: ys, mode: 'markers', type: 'scattergl', marker: { size: 4, color: COLORS.slate, opacity: 0.4 }, name: '測試樣本' },
      { x: grid, y: band.map(b => b.piHi), mode: 'lines', line: { color: COLORS.teal, dash: 'dot' }, name: `${conf}% PI 上界` },
      { x: grid, y: band.map(b => b.piLo), mode: 'lines', line: { color: COLORS.teal, dash: 'dot' }, name: `${conf}% PI 下界` },
      { x: grid, y: band.map(b => b.ciHi), mode: 'lines', line: { width: 0 }, showlegend: false, hoverinfo: 'skip' },
      { x: grid, y: band.map(b => b.ciLo), mode: 'lines', line: { width: 0 }, fill: 'tonexty', fillcolor: 'rgba(225,29,72,0.35)', name: `${conf}% CI` },
      { x: grid, y: band.map(b => b.yhat), mode: 'lines', line: { color: COLORS.rose, width: 2.5 }, name: '回歸線' },
    ], baseLayout(`簡單線性回歸：${fname} → ${yLab} (R² = ${fmt(slr.r2, 4)})，含 ${conf}% CI 與 PI`, fname, yLab), PLOT_CFG);
  }

  // ------------------------------------------------------------------ Deployment
  function renderDeployment(D, R) {
    const used = new Set(R.fit.kept.map(j => D.features[j]));
    const inputs = [];
    for (const g of D.groups) {
      if (!g.cols.some(c => used.has(c))) continue;
      if (g.type === 'num') {
        const j = D.features.indexOf(g.name);
        const vals = R.train.map(i => D.X[i][j]);
        const med = Stats.median(vals);
        inputs.push(`<label class="field"><span>${esc(g.name)}</span><input type="number" step="any" data-num="${j}" value="${+med.toPrecision(6)}"><small>範圍 ${fmt(Math.min(...vals))} ~ ${fmt(Math.max(...vals))}</small></label>`);
      } else {
        inputs.push(`<label class="field"><span>${esc(g.name)}</span><select data-cat="${esc(g.name)}">${g.levels.map(l => `<option>${esc(l)}</option>`).join('')}</select><small>類別變數</small></label>`);
      }
    }
    $('deployForm').innerHTML = inputs.join('') || '<p class="muted">目前模型沒有任何特徵。</p>';
    $('deployForm').querySelectorAll('input,select').forEach(el => el.addEventListener('input', predictOne));
    predictOne();
  }

  function predictOne() {
    const D = S.data, R = S.result;
    if (!D || !R) return;
    const x = Float64Array.from(R.mu); // 未使用的特徵預設為平均值 (不影響預測)
    $('deployForm').querySelectorAll('[data-num]').forEach(el => { x[+el.dataset.num] = Number(el.value); });
    $('deployForm').querySelectorAll('[data-cat]').forEach(el => {
      const g = D.groups.find(gg => gg.name === el.dataset.cat);
      g.cols.forEach(c => { x[D.features.indexOf(c)] = (c === `${g.name}_${el.value}`) ? 1 : 0; });
    });
    const z = Array.from(x, (v, j) => (v - R.mu[j]) / R.sd[j]);
    const { yhat, lev } = Stats.predictZ(R.fit, z);
    const seF = Math.sqrt(R.fit.sigma2 * lev), seP = Math.sqrt(R.fit.sigma2 * (1 + lev));
    const b = R.back, tc = R.tc, conf = Math.round(R.cfg.conf * 100);
    const out = { yhat: b(yhat), ciLo: b(yhat - tc * seF), ciHi: b(yhat + tc * seF), piLo: b(yhat - tc * seP), piHi: b(yhat + tc * seP) };
    $('deployResult').innerHTML = `
      <div class="pred-main">預測 ${esc(D.target)}：<b>${fmt(out.yhat)}</b></div>
      <div>${conf}% 信賴區間 (平均值)：[${fmt(out.ciLo)}, ${fmt(out.ciHi)}]</div>
      <div>${conf}% 預測區間 (個別值)：[${fmt(out.piLo)}, ${fmt(out.piHi)}]</div>`;
    Plotly.react('chartDeploy', [
      { x: [out.piLo, out.piHi], y: ['PI', 'PI'], mode: 'lines', line: { color: COLORS.teal, width: 14 }, name: `${conf}% PI` },
      { x: [out.ciLo, out.ciHi], y: ['CI', 'CI'], mode: 'lines', line: { color: COLORS.primary, width: 14 }, name: `${conf}% CI` },
      { x: [out.yhat, out.yhat], y: ['CI', 'PI'], mode: 'markers', marker: { size: 16, color: COLORS.amber, symbol: 'diamond' }, name: '點估計' },
    ], { ...baseLayout('單筆預測之點估計與區間', D.target, ''), height: 220, showlegend: false, margin: { l: 50, r: 20, t: 40, b: 40 } }, PLOT_CFG);
  }

  function downloadPreds() {
    const D = S.data, R = S.result;
    if (!R) return;
    const conf = Math.round(R.cfg.conf * 100);
    const head = ['row', D.target, 'predicted', `ci${conf}_low`, `ci${conf}_high`, `pi${conf}_low`, `pi${conf}_high`];
    const lines = [head.join(',')].concat(R.preds.map(q => [q.i, q.y, q.yhat, q.ciLo, q.ciHi, q.piLo, q.piHi].join(',')));
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = 'test_predictions.csv'; a.click();
  }

  // ------------------------------------------------------------------ 共用
  function kpis(list, raw) {
    const html = list.map(([k, v]) => `<div class="kpi"><div class="k">${k}</div><div class="v">${v}</div></div>`).join('');
    return raw ? html : html;
  }
  function baseLayout(title, xt, yt) {
    return {
      title: { text: title, font: { size: 14 } }, xaxis: { title: xt, zeroline: false }, yaxis: { title: yt, zeroline: false },
      margin: { l: 60, r: 20, t: 50, b: 50 }, height: 420, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: '#fff',
      legend: { orientation: 'h', y: -0.2 }, font: { family: 'Noto Sans TC, Microsoft JhengHei, sans-serif' },
    };
  }
  function sample(arr, m, seed) {
    if (arr.length <= m) return arr.slice();
    return Stats.shuffle(arr.length, seed).slice(0, m).map(i => arr[i]);
  }

  // ------------------------------------------------------------------ 事件綁定
  function init() {
    const sel = $('problemSelect');
    sel.innerHTML = Object.entries(PROBLEMS).map(([k, p]) => `<option value="${k}">${p.label}</option>`).join('');
    $('examples').innerHTML = EXAMPLES.map(e => `<button class="chip" data-url="${e.url}">${esc(e.name)}</button>`).join('');
    $('examples').addEventListener('click', ev => {
      const b = ev.target.closest('[data-url]'); if (!b) return;
      $('urlInput').value = b.dataset.url; loadCustomUrl();
    });
    sel.addEventListener('change', () => loadProblem(sel.value));
    $('loadUrlBtn').addEventListener('click', loadCustomUrl);
    $('urlInput').addEventListener('keydown', e => { if (e.key === 'Enter') loadCustomUrl(); });
    $('fileInput').addEventListener('change', e => { if (e.target.files[0]) loadCustomFile(e.target.files[0]); });
    $('targetSelect').addEventListener('change', () => prepareAndRun($('targetSelect').value, true));
    $('kSlider').addEventListener('input', () => { $('kVal').textContent = $('kSlider').value; });
    $('testSlider').addEventListener('input', () => { $('testVal').textContent = $('testSlider').value + '%'; });
    ['methodSelect', 'critSelect', 'alphaSelect', 'confSelect', 'logY', 'kSlider', 'testSlider', 'seedInput'].forEach(id =>
      $(id).addEventListener('change', () => { syncMethodUI(); run(); }));
    $('poolBox').addEventListener('change', () => {
      $('poolCount').textContent = `${getPool().length} / ${S.data ? S.data.features.length : 0}`; run();
    });
    $('poolAll').addEventListener('click', () => { $('poolBox').querySelectorAll('input').forEach(b => { b.checked = true; }); $('poolBox').dispatchEvent(new Event('change')); });
    $('poolNone').addEventListener('click', () => { $('poolBox').querySelectorAll('input').forEach(b => { b.checked = false; }); $('poolCount').textContent = `0 / ${S.data ? S.data.features.length : 0}`; });
    $('dlBtn').addEventListener('click', downloadPreds);
    syncMethodUI();
    // 支援網址參數 ?problem=flood 或 ?url=...
    const qs = new URLSearchParams(location.search);
    if (qs.get('url')) {
      sel.value = 'custom'; loadProblem('custom'); $('urlInput').value = qs.get('url'); loadCustomUrl();
    } else {
      const k = PROBLEMS[qs.get('problem')] ? qs.get('problem') : 'kc';
      sel.value = k; loadProblem(k);
    }
  }

  function syncMethodUI() {
    const m = $('methodSelect').value;
    $('critRow').classList.toggle('dim', m !== 'forward');
    $('kRow').classList.toggle('dim', m !== 'kbest');
    $('alphaRow').classList.toggle('dim', m !== 'backward');
  }

  document.addEventListener('DOMContentLoaded', init);
})();
