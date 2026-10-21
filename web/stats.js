/* =============================================================================
 * stats.js — 純前端 (瀏覽器端) 多元線性回歸統計引擎
 * -----------------------------------------------------------------------------
 *  - OLS 以「充分統計量 (Gram matrix)」求解：任何特徵子集都可在 O(k^3) 內重新擬合，
 *    讓逐步選擇 (Forward / Backward)、K-Fold CV 都能即時在瀏覽器完成。
 *  - 提供 t 分配 CDF / 反函數 (p-value、信賴區間、預測區間)、常態反函數 (Q-Q plot)。
 *  - 自動偵測完全共線性 (aliased columns，例如 sqft_living = sqft_above + sqft_basement)。
 * ===========================================================================*/
const Stats = (() => {
  // ---------------------------------------------------------------- 分配函數
  function logGamma(x) {
    const c = [76.18009172947146, -86.50532032941677, 24.01409824083091,
      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5];
    let y = x, tmp = x + 5.5;
    tmp -= (x + 0.5) * Math.log(tmp);
    let ser = 1.000000000190015;
    for (let j = 0; j < 6; j++) ser += c[j] / ++y;
    return -tmp + Math.log(2.5066282746310005 * ser / x);
  }

  function betacf(a, b, x) {
    const MAXIT = 300, EPS = 3e-14, FPMIN = 1e-300;
    const qab = a + b, qap = a + 1, qam = a - 1;
    let c = 1, d = 1 - qab * x / qap;
    if (Math.abs(d) < FPMIN) d = FPMIN;
    d = 1 / d;
    let h = d;
    for (let m = 1; m <= MAXIT; m++) {
      const m2 = 2 * m;
      let aa = m * (b - m) * x / ((qam + m2) * (a + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d; h *= d * c;
      aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
      d = 1 + aa * d; if (Math.abs(d) < FPMIN) d = FPMIN;
      c = 1 + aa / c; if (Math.abs(c) < FPMIN) c = FPMIN;
      d = 1 / d;
      const del = d * c;
      h *= del;
      if (Math.abs(del - 1) < EPS) break;
    }
    return h;
  }

  /** 正規化不完全 Beta 函數 I_x(a, b) */
  function ibeta(x, a, b) {
    if (x <= 0) return 0;
    if (x >= 1) return 1;
    const bt = Math.exp(logGamma(a + b) - logGamma(a) - logGamma(b) +
      a * Math.log(x) + b * Math.log(1 - x));
    if (x < (a + 1) / (a + b + 2)) return bt * betacf(a, b, x) / a;
    return 1 - bt * betacf(b, a, 1 - x) / b;
  }

  /** Student t 雙尾 p-value */
  function tPValue(t, df) {
    if (!isFinite(t)) return 0;
    return ibeta(df / (df + t * t), df / 2, 0.5);
  }

  /** F 分配右尾 p-value：P(F > f) */
  function fPValue(f, d1, d2) {
    if (!isFinite(f)) return 0;
    if (f <= 0) return 1;
    return ibeta(d2 / (d2 + d1 * f), d2 / 2, d1 / 2);
  }

  function tCdf(t, df) {
    const p = 0.5 * ibeta(df / (df + t * t), df / 2, 0.5);
    return t > 0 ? 1 - p : p;
  }

  /** Student t 反函數 (分位數)，二分法 */
  function tInv(p, df) {
    if (df > 1e6) return normInv(p);
    let lo = -1e3, hi = 1e3;
    for (let i = 0; i < 200; i++) {
      const mid = (lo + hi) / 2;
      if (tCdf(mid, df) < p) lo = mid; else hi = mid;
    }
    return (lo + hi) / 2;
  }

  /** 標準常態反函數 (Acklam 近似) */
  function normInv(p) {
    const a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    const b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01];
    const c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    const d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00];
    const pl = 0.02425;
    let q, r;
    if (p < pl) {
      q = Math.sqrt(-2 * Math.log(p));
      return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
        ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (p <= 1 - pl) {
      q = p - 0.5; r = q * q;
      return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }
    q = Math.sqrt(-2 * Math.log(1 - p));
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }

  // ---------------------------------------------------------------- 線性代數
  /** Gauss-Jordan 反矩陣 (部分樞軸) */
  function invert(A) {
    const n = A.length;
    const M = A.map((row, i) => {
      const r = new Float64Array(2 * n);
      for (let j = 0; j < n; j++) r[j] = row[j];
      r[n + i] = 1;
      return r;
    });
    for (let col = 0; col < n; col++) {
      let piv = col, best = Math.abs(M[col][col]);
      for (let r = col + 1; r < n; r++) {
        const v = Math.abs(M[r][col]);
        if (v > best) { best = v; piv = r; }
      }
      if (best < 1e-300) throw new Error('Singular matrix');
      if (piv !== col) { const t = M[piv]; M[piv] = M[col]; M[col] = t; }
      const d = M[col][col];
      for (let j = 0; j < 2 * n; j++) M[col][j] /= d;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const f = M[r][col];
        if (f === 0) continue;
        for (let j = 0; j < 2 * n; j++) M[r][j] -= f * M[col][j];
      }
    }
    return M.map(r => Array.from(r.slice(n)));
  }

  /**
   * 依序做 Cholesky 分解並略過「與前面欄位完全線性相依」的欄位。
   * 回傳保留下來的索引 (相對於 idx)。
   */
  function nonAliased(C, idx, tol = 1e-9) {
    const kept = [], L = [];
    for (const j of idx) {
      const v = new Array(kept.length);
      let ss = 0;
      for (let t = 0; t < kept.length; t++) {
        let s = C[j][kept[t]];
        for (let u = 0; u < t; u++) s -= L[t][u] * v[u];
        v[t] = s / L[t][t];
        ss += v[t] * v[t];
      }
      const r = C[j][j] - ss;
      if (C[j][j] > 0 && r > tol * C[j][j]) {
        v.push(Math.sqrt(r));
        L.push(v);
        kept.push(j);
      }
    }
    return kept;
  }

  // ---------------------------------------------------------------- 充分統計量
  /**
   * 對 Z (n x p，列為 Float64Array) 的部分列計算 Σz、Σy、Z'Z、Z'y、y'y
   */
  function gram(Z, y, rows) {
    const p = Z[0].length;
    const zz = Array.from({ length: p }, () => new Float64Array(p));
    const sz = new Float64Array(p), zy = new Float64Array(p);
    let sy = 0, yy = 0;
    for (const i of rows) {
      const z = Z[i], yi = y[i];
      sy += yi; yy += yi * yi;
      for (let a = 0; a < p; a++) {
        const za = z[a];
        sz[a] += za; zy[a] += za * yi;
        const row = zz[a];
        for (let b = a; b < p; b++) row[b] += za * z[b];
      }
    }
    for (let a = 0; a < p; a++) for (let b = 0; b < a; b++) zz[a][b] = zz[b][a];
    return { n: rows.length, sz, sy, zz, zy, yy, p };
  }

  function subtractGram(A, B) {
    const p = A.p;
    return {
      n: A.n - B.n, p, sy: A.sy - B.sy, yy: A.yy - B.yy,
      sz: A.sz.map((v, i) => v - B.sz[i]), zy: A.zy.map((v, i) => v - B.zy[i]),
      zz: A.zz.map((r, i) => r.map((v, j) => v - B.zz[i][j])),
    };
  }

  /** 由充分統計量計算中心化的交叉乘積 */
  function center(G) {
    const n = G.n, p = G.p;
    const mz = Array.from(G.sz, v => v / n), my = G.sy / n;
    const C = Array.from({ length: p }, (_, a) => {
      const r = new Float64Array(p);
      for (let b = 0; b < p; b++) r[b] = G.zz[a][b] - n * mz[a] * mz[b];
      return r;
    });
    const c = Array.from({ length: p }, (_, a) => G.zy[a] - n * mz[a] * my);
    const syy = G.yy - n * my * my;
    return { n, p, mz, my, C, c, syy };
  }

  /**
   * 以中心化統計量擬合指定特徵子集 S (在 z-space)。
   * 回傳 b (z-space 斜率)、截距、(Z'Z)^-1、SSE、σ² 等
   */
  function fitSubset(K, S) {
    const kept = S.length ? nonAliased(K.C, S) : [];
    const aliased = S.filter(j => !kept.includes(j));
    const k = kept.length;
    let b = [], Cinv = [], reg = 0;
    if (k > 0) {
      const sub = kept.map(a => kept.map(bb => K.C[a][bb]));
      Cinv = invert(sub);
      const cs = kept.map(a => K.c[a]);
      b = Cinv.map(row => row.reduce((s, v, j) => s + v * cs[j], 0));
      reg = b.reduce((s, v, j) => s + v * cs[j], 0);
    }
    const sse = Math.max(K.syy - reg, 1e-300);
    const df = K.n - k - 1;
    const sigma2 = sse / Math.max(df, 1);
    const intercept = K.my - b.reduce((s, v, j) => s + v * K.mz[kept[j]], 0);
    const r2 = 1 - sse / K.syy;
    const adjR2 = 1 - (1 - r2) * (K.n - 1) / Math.max(df, 1);
    const aic = K.n * Math.log(sse / K.n) + 2 * (k + 1);
    const bic = K.n * Math.log(sse / K.n) + Math.log(K.n) * (k + 1);
    return { kept, aliased, k, b, Cinv, intercept, sse, df, sigma2, r2, adjR2, aic, bic, n: K.n, mz: K.mz };
  }

  /** z-space 預測 (含點估計 + 預測變異數的「leverage」部分) */
  function predictZ(fit, z) {
    let yhat = fit.intercept;
    const k = fit.k;
    const d = new Array(k);
    for (let j = 0; j < k; j++) {
      yhat += fit.b[j] * z[fit.kept[j]];
      d[j] = z[fit.kept[j]] - fit.mz[fit.kept[j]];
    }
    let q = 0;
    for (let a = 0; a < k; a++) {
      let s = 0;
      const row = fit.Cinv[a];
      for (let bb = 0; bb < k; bb++) s += row[bb] * d[bb];
      q += d[a] * s;
    }
    const lev = 1 / fit.n + q; // x0'(X'X)^-1 x0
    return { yhat, lev };
  }

  // ---------------------------------------------------------------- 工具
  function mean(a) { let s = 0; for (const v of a) s += v; return s / a.length; }
  function std(a) {
    const m = mean(a); let s = 0;
    for (const v of a) s += (v - m) ** 2;
    return Math.sqrt(s / a.length);
  }
  function median(a) {
    const s = Float64Array.from(a).sort();
    const m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }
  function quantile(a, q) {
    const s = Float64Array.from(a).sort();
    const pos = (s.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return s[lo] + (s[hi] - s[lo]) * (pos - lo);
  }
  function skew(a) {
    const m = mean(a), sd = std(a); let s = 0;
    for (const v of a) s += ((v - m) / sd) ** 3;
    return s / a.length;
  }
  function corr(x, y) {
    const mx = mean(x), my = mean(y);
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < x.length; i++) {
      const dx = x[i] - mx, dy = y[i] - my;
      sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
  }

  /** 可重現的亂數 (mulberry32) */
  function rng(seed) {
    return function () {
      seed |= 0; seed = seed + 0x6D2B79F5 | 0;
      let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function shuffle(n, seed) {
    const r = rng(seed), idx = Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(r() * (i + 1));
      [idx[i], idx[j]] = [idx[j], idx[i]];
    }
    return idx;
  }

  function metrics(y, yhat, k) {
    const n = y.length, my = mean(y);
    let sse = 0, sst = 0, sae = 0, sape = 0, nape = 0;
    for (let i = 0; i < n; i++) {
      const e = y[i] - yhat[i];
      sse += e * e; sst += (y[i] - my) ** 2; sae += Math.abs(e);
      if (y[i] !== 0) { sape += Math.abs(e / y[i]); nape++; }
    }
    const r2 = 1 - sse / sst;
    return {
      r2, adjR2: 1 - (1 - r2) * (n - 1) / Math.max(n - k - 1, 1),
      rmse: Math.sqrt(sse / n), mae: sae / n, mape: nape ? 100 * sape / nape : NaN, mse: sse / n,
    };
  }

  return {
    tPValue, fPValue, tInv, tCdf, normInv, invert, nonAliased, gram, subtractGram, center,
    fitSubset, predictZ, mean, std, median, quantile, skew, corr, shuffle, metrics,
  };
})();
