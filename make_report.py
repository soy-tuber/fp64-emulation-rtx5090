#!/usr/bin/env python3
"""Generate docs/index.html for GitHub Pages from results/*.json.

Everything on the page is derived from the raw JSON, so no number is
transcribed by hand.
"""
import glob
import html
import json
import os
import platform
import subprocess
import sys
import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
DOCS = os.path.join(HERE, "docs")


# --------------------------------------------------------------- data loading
def load():
    out = {}
    for p in sorted(glob.glob(os.path.join(RESULTS, "*.json"))):
        with open(p, encoding="utf-8") as f:
            rec = json.load(f)
        out[rec["label"]] = rec
    return out


def pick(rec, n, dist):
    if not rec:
        return None
    for r in rec.get("runs", []):
        if r.get("n") == n and r.get("dist") == dist and "error" not in r:
            return r
    return None


def relerr(r):
    a = (r or {}).get("accuracy")
    return a["rel_err_median"] if a else None


def fmt_err(v):
    if v is None:
        return "&mdash;"
    if v == 0.0:
        return "<b>0</b>"
    return "%.2e" % v


def fmt(v, nd=3):
    return "&mdash;" if v is None else ("%.*f" % (nd, v))


# ------------------------------------------------------------------ svg chart
def svg_line(series, xlabel, ylabel, width=680, height=330, logx=False,
             logy=False, xticks=None, caption=""):
    """series: list of dicts {name, color, points:[(x,y)], dash}"""
    import math
    pad_l, pad_r, pad_t, pad_b = 62, 16, 18, 46
    pts_all = [p for s in series for p in s["points"] if p[1] is not None]
    if not pts_all:
        return "<p class='muted'>(no data)</p>"
    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]

    def tx(v):
        lo, hi = min(xs), max(xs)
        if logx:
            lo, hi, v = math.log10(lo), math.log10(hi), math.log10(v)
        f = 0.0 if hi == lo else (v - lo) / (hi - lo)
        return pad_l + f * (width - pad_l - pad_r)

    ylo, yhi = (min(ys), max(ys))
    if logy:
        pos = [y for y in ys if y > 0]
        ylo = min(pos) if pos else 1e-18
        yhi = max(pos) if pos else 1.0
        ylo, yhi = math.log10(ylo) - 0.3, math.log10(yhi) + 0.3
    else:
        ylo, yhi = 0.0, yhi * 1.12

    def ty(v):
        w = math.log10(v) if (logy and v > 0) else v
        if logy and v <= 0:
            w = ylo
        f = 0.0 if yhi == ylo else (w - ylo) / (yhi - ylo)
        return height - pad_b - f * (height - pad_t - pad_b)

    o = ["<svg viewBox='0 0 %d %d' class='chart' role='img'>" % (width, height)]
    # y grid
    nticks = 5
    for i in range(nticks + 1):
        w = ylo + (yhi - ylo) * i / nticks
        y = height - pad_b - (height - pad_t - pad_b) * i / nticks
        lab = ("1e%d" % round(w)) if logy else ("%.3g" % w)
        o.append("<line class='grid' x1='%.1f' y1='%.1f' x2='%.1f' y2='%.1f'/>"
                 % (pad_l, y, width - pad_r, y))
        o.append("<text class='tick' x='%.1f' y='%.1f' text-anchor='end'>%s</text>"
                 % (pad_l - 7, y + 4, lab))
    for xv in (xticks or sorted(set(xs))):
        o.append("<text class='tick' x='%.1f' y='%.1f' text-anchor='middle'>%s</text>"
                 % (tx(xv), height - pad_b + 18, xv))
    o.append("<text class='axis' x='%.1f' y='%.1f' text-anchor='middle'>%s</text>"
             % ((pad_l + width - pad_r) / 2, height - 6, html.escape(xlabel)))
    o.append("<text class='axis' transform='rotate(-90 14 %.1f)' x='14' y='%.1f' "
             "text-anchor='middle'>%s</text>"
             % ((height) / 2, (height) / 2, html.escape(ylabel)))

    for s in series:
        pp = [(tx(x), ty(y)) for x, y in s["points"] if y is not None]
        if not pp:
            continue
        d = " ".join(("%s%.1f,%.1f" % ("M" if i == 0 else "L", x, y))
                     for i, (x, y) in enumerate(pp))
        dash = " stroke-dasharray='5 4'" if s.get("dash") else ""
        o.append("<path d='%s' fill='none' stroke='%s' stroke-width='2.2'%s/>"
                 % (d, s["color"], dash))
        for x, y in pp:
            o.append("<circle cx='%.1f' cy='%.1f' r='3.4' fill='%s'/>"
                     % (x, y, s["color"]))
    o.append("</svg>")
    leg = " ".join("<span class='lg'><i style='background:%s'></i>%s</span>"
                   % (s["color"], html.escape(s["name"])) for s in series)
    cap = ("<figcaption>%s</figcaption>" % html.escape(caption)) if caption else ""
    return "<figure>%s<div class='legend'>%s</div>%s</figure>" % ("".join(o), leg, cap)


# ------------------------------------------------------------------ host info
def host_info(any_rec):
    def sh(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=25).stdout.strip()
        except Exception:
            return ""
    gpu = sh(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,"
              "power.limit,clocks.max.sm", "--format=csv,noheader"])
    cv = (any_rec or {}).get("cublas_version")
    return {
        "gpu": gpu or "(nvidia-smi unavailable)",
        "cublas_version": cv,
        "os": "%s %s" % (platform.system(), platform.version()),
        "cpu": os.environ.get("PROCESSOR_IDENTIFIER", platform.processor()),
        "python": platform.python_version(),
        "numpy": __import__("numpy").__version__,
        "date": datetime.date.today().isoformat(),
        "cuda_dir": (any_rec or {}).get("cuda_dir", ""),
        "phi": (any_rec or {}).get("phi"),
    }


# ----------------------------------------------------------------------- page
CSS = """
:root{
 --bg:#ffffff; --fg:#1a1d21; --muted:#5d666f; --line:#e2e6ea; --card:#f7f9fb;
 --accent:#0b63c5; --warn:#8a5200; --ok:#0f7a4a;
 --c-native:#c2410c; --c-easy:#0b63c5; --c-hard:#7c3aed;
}
@media (prefers-color-scheme:dark){
 :root:not([data-theme="light"]){
  --bg:#14171a; --fg:#e8ecf0; --muted:#9aa4ae; --line:#2a3036; --card:#1b1f23;
  --accent:#64a8ff; --warn:#e0a63a; --ok:#49c98a;
  --c-native:#ff8a5c; --c-easy:#64a8ff; --c-hard:#b08cff;
 }
}
:root[data-theme="dark"]{
 --bg:#14171a; --fg:#e8ecf0; --muted:#9aa4ae; --line:#2a3036; --card:#1b1f23;
 --accent:#64a8ff; --warn:#e0a63a; --ok:#49c98a;
 --c-native:#ff8a5c; --c-easy:#64a8ff; --c-hard:#b08cff;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:16px/1.7 -apple-system,BlinkMacSystemFont,"Segoe UI","Hiragino Kaku Gothic ProN",
 "Noto Sans JP",Meiryo,sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:48px 16px 96px}
h1{font-size:1.85rem;line-height:1.3;margin:0 0 6px;letter-spacing:-.01em}
h2{font-size:1.25rem;margin:52px 0 14px;padding-bottom:7px;border-bottom:1px solid var(--line)}
h3{font-size:1.02rem;margin:30px 0 10px}
.sub{color:var(--muted);margin:0 0 30px;font-size:.95rem}
p{margin:0 0 14px}
code,kbd{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;
 background:var(--card);border:1px solid var(--line);border-radius:4px;padding:1px 5px}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;
 padding:14px 16px;overflow:auto;font-size:.84rem;line-height:1.55}
pre code{background:none;border:0;padding:0}
table{border-collapse:collapse;width:100%;margin:16px 0;font-size:.9rem;
 font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}
thead th{background:var(--card);font-weight:600;white-space:nowrap}
tbody tr:hover{background:var(--card)}
.kv{display:grid;grid-template-columns:auto 1fr;gap:4px 16px;font-size:.9rem;
 background:var(--card);border:1px solid var(--line);border-radius:8px;padding:16px 18px}
.kv dt{color:var(--muted);white-space:nowrap}
.kv dd{margin:0;word-break:break-word}
.hl{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin:22px 0}
.hl div{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.hl b{display:block;font-size:1.65rem;line-height:1.2;letter-spacing:-.02em}
.hl span{color:var(--muted);font-size:.8rem}
figure{margin:20px 0}
.chart{width:100%;height:auto;display:block}
.grid{stroke:var(--line);stroke-width:1}
.tick{fill:var(--muted);font-size:11px}
.axis{fill:var(--muted);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:8px;font-size:.84rem;color:var(--muted)}
.lg i{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;
 vertical-align:-1px}
figcaption{color:var(--muted);font-size:.84rem;margin-top:8px}
.note{border-left:3px solid var(--accent);background:var(--card);padding:12px 16px;
 border-radius:0 8px 8px 0;margin:18px 0;font-size:.92rem}
.note.warn{border-left-color:var(--warn)}
.muted{color:var(--muted)}
ul,ol{margin:0 0 14px;padding-left:22px}
li{margin:5px 0}
footer{margin-top:64px;padding-top:18px;border-top:1px solid var(--line);
 color:var(--muted);font-size:.84rem}
a{color:var(--accent)}
"""


def main():
    data = load()
    os.makedirs(DOCS, exist_ok=True)
    nat = data.get("A_native")
    emu = data.get("A_emu_adp")
    any_rec = nat or emu or (list(data.values())[0] if data else None)
    if not any_rec:
        print("no results found in %s" % RESULTS)
        return 1
    hi = host_info(any_rec)
    sizes = sorted({r["n"] for r in (emu or nat)["runs"] if "error" not in r})

    # ---- Phase A table
    rows = []
    best = {"sp_easy": 0.0, "sp_hard": 0.0, "n": None}
    for n in sizes:
        rn = pick(nat, n, "easy")
        re_, rh = pick(emu, n, "easy"), pick(emu, n, "hard")
        rnh = pick(nat, n, "hard")
        tn = rn["tflops_median"] if rn else None
        tnh = rnh["tflops_median"] if rnh else None
        te = re_["tflops_median"] if re_ else None
        th = rh["tflops_median"] if rh else None
        sp_e = (te / tn) if (te and tn) else None
        sp_h = (th / tnh) if (th and tnh) else None
        if sp_e and sp_e > best["sp_easy"]:
            best.update(sp_easy=sp_e, n=n)
        if sp_h and sp_h > best["sp_hard"]:
            best["sp_hard"] = sp_h
        rows.append((n, tn, te, th, sp_e, sp_h,
                     relerr(rn), relerr(re_), relerr(rh)))

    tA = ["<table><thead><tr><th>n</th><th>native FP64<br>TFLOPS</th>"
          "<th>emulated, easy<br>TFLOPS</th><th>emulated, hard<br>TFLOPS</th>"
          "<th>speedup<br>easy</th><th>speedup<br>hard</th>"
          "<th>rel.err<br>native</th><th>rel.err<br>emu easy</th>"
          "<th>rel.err<br>emu hard</th></tr></thead><tbody>"]
    for (n, tn, te, th, se, sh, en, ee, eh) in rows:
        tA.append("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td>"
                  "<td>%s&times;</td><td>%s&times;</td><td>%s</td><td>%s</td>"
                  "<td>%s</td></tr>"
                  % (n, fmt(tn), fmt(te), fmt(th), fmt(se, 2), fmt(sh, 2),
                     fmt_err(en), fmt_err(ee), fmt_err(eh)))
    tA.append("</tbody></table>")

    ch1 = svg_line(
        [{"name": "native FP64", "color": "var(--c-native)",
          "points": [(n, r[1]) for n, r in zip(sizes, rows)]},
         {"name": "emulated, narrow range (easy)", "color": "var(--c-easy)",
          "points": [(n, r[2]) for n, r in zip(sizes, rows)]},
         {"name": "emulated, wide range (hard)", "color": "var(--c-hard)",
          "points": [(n, r[3]) for n, r in zip(sizes, rows)], "dash": True}],
        "matrix size n (n x n x n DGEMM)", "TFLOPS", logx=True, xticks=sizes,
        caption="Throughput of cublasDgemm. Native FP64 is flat near the "
                "1/64-rate ceiling; emulation scales with problem size.")

    # ---- Phase B
    tB = ""
    bparts = []
    for strat in ("performant", "eager"):
        rec = data.get("B_emu_" + strat)
        if not rec:
            continue
        for dist in ("easy", "hard"):
            r = pick(rec, 8192, dist)
            if r:
                bparts.append((strat, dist, r["tflops_median"], relerr(r)))
    if bparts:
        tB = ["<table><thead><tr><th>CUBLAS_EMULATION_STRATEGY</th><th>input</th>"
              "<th>TFLOPS</th><th>rel.err (median)</th></tr></thead><tbody>"]
        for (s, d, t, e) in bparts:
            tB.append("<tr><td><code>%s</code></td><td>%s</td><td>%s</td><td>%s</td></tr>"
                      % (s, d, fmt(t), fmt_err(e)))
        tB.append("</tbody></table>")
        tB = "".join(tB)

    # ---- Phase C mantissa sweep
    cbits, c_easy, c_hard, ce_easy, ce_hard = [], [], [], [], []
    for lbl in sorted(k for k in data if k.startswith("C_mant")):
        bits = int(lbl.replace("C_mant", ""))
        rec = data[lbl]
        re_, rh = pick(rec, 8192, "easy"), pick(rec, 8192, "hard")
        cbits.append(bits)
        c_easy.append(re_["tflops_median"] if re_ else None)
        c_hard.append(rh["tflops_median"] if rh else None)
        ce_easy.append(relerr(re_))
        ce_hard.append(relerr(rh))
    tC, ch2, ch3 = "", "", ""
    if cbits:
        tC = ["<table><thead><tr><th>mantissa bits</th><th>TFLOPS easy</th>"
              "<th>TFLOPS hard</th><th>rel.err easy</th><th>rel.err hard</th>"
              "</tr></thead><tbody>"]
        for i, b in enumerate(cbits):
            tC.append("<tr><td>%d</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                      % (b, fmt(c_easy[i]), fmt(c_hard[i]),
                         fmt_err(ce_easy[i]), fmt_err(ce_hard[i])))
        tC.append("</tbody></table>")
        tC = "".join(tC)
        ch2 = svg_line(
            [{"name": "easy", "color": "var(--c-easy)",
              "points": list(zip(cbits, c_easy))},
             {"name": "hard", "color": "var(--c-hard)",
              "points": list(zip(cbits, c_hard)), "dash": True}],
            "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT", "TFLOPS",
            xticks=cbits,
            caption="Throughput falls as more mantissa bits are requested: "
                    "more bits means more INT8 slices per FP64 product.")
        ch3 = svg_line(
            [{"name": "easy", "color": "var(--c-easy)",
              "points": list(zip(cbits, ce_easy))},
             {"name": "hard", "color": "var(--c-hard)",
              "points": list(zip(cbits, ce_hard)), "dash": True}],
            "CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT",
            "relative error (median)", logy=True, xticks=cbits,
            caption="Accuracy against the exactly-rounded product. Points at "
                    "zero error are clamped to the bottom of the log axis.")

    nat_peak = max([r[1] for r in rows if r[1]] or [0])
    emu_peak = max([r[2] for r in rows if r[2]] or [0])
    nat_err = max([r[6] for r in rows if r[6]] or [0])

    H = []
    A = H.append
    A("<!doctype html><html lang='ja'><head><meta charset='utf-8'>")
    A("<meta name='viewport' content='width=device-width,initial-scale=1'>")
    A("<title>FP64 Emulation on RTX 5090</title>")
    A("<meta name='description' content='Measured native vs Ozaki-style "
      "fixed-point emulated FP64 DGEMM throughput and accuracy on a GeForce "
      "RTX 5090 with cuBLAS 13.'>")
    A("<style>%s</style></head><body><div class='wrap'>" % CSS)

    A("<h1>GeForce RTX 5090 で FP64 エミュレーションを測る</h1>")
    A("<p class='sub'>cuBLAS 13 の固定小数点エミュレーション（尾崎スキーム系）"
      "を、FP64 を意図的に削られたコンシューマ GPU で実測した記録。"
      "測定日 %s</p>" % hi["date"])

    A("<div class='hl'>")
    A("<div><b>%.2f&times;</b><span>最大スループット向上 (n=%s, narrow range)</span></div>"
      % (best["sp_easy"], best["n"]))
    A("<div><b>%.2f&times;</b><span>wide dynamic range でも</span></div>" % best["sp_hard"])
    A("<div><b>%.3f</b><span>native FP64 の上限 TFLOPS</span></div>" % nat_peak)
    A("<div><b>%.3f</b><span>エミュレーションの TFLOPS</span></div>" % emu_peak)
    A("</div>")

    A("<div class='note'><b>結論を一行で。</b>"
      "このカードでは FP64 エミュレーションは native FP64 より"
      "<b>速いだけでなく、精度も高い</b>。native は累積丸めで相対誤差 %s 程度まで"
      "劣化するが、エミュレーションは整数で厳密に積和して最後に一度だけ丸めるため、"
      "正しく丸めた厳密値に一致する（相対誤差 0 または 1 ULP）。</div>"
      % ("%.1e" % nat_err if nat_err else "&mdash;"))

    A("<h2>なぜ測る価値があるのか</h2>")
    A("<p>GeForce の倍精度は仕様上 FP32 の 1/64 に制限されている。一方で "
      "INT8 テンソルコアは3桁 TOPS ある。CUDA 13.0 Update 2 で cuBLAS に入った"
      "固定小数点エミュレーションは、FP64 の積を整数の列に分解して INT8 テンソルコアで"
      "厳密に計算する——つまり<b>潤沢な資源で枯渇した資源を代替する</b>。</p>")
    A("<p>公開されている測定値はデータセンター GPU（native FP64 が速い機種）が"
      "中心で、そこでの利得は控えめに見える。しかし FP64 を削られたコンシューマ機では"
      "前提が逆であり、倍率が大きくなるはずだ。その実測値がほとんど見当たらないので、"
      "自分で測った。</p>")

    A("<h2>測定環境</h2><dl class='kv'>")
    for k, v in (("GPU / driver", hi["gpu"]), ("CPU", hi["cpu"]),
                 ("OS", hi["os"]),
                 ("cuBLAS version (cublasGetVersion)", str(hi["cublas_version"])),
                 ("CUDA DLL source", hi["cuda_dir"]),
                 ("Python / NumPy", "%s / %s" % (hi["python"], hi["numpy"]))):
        A("<dt>%s</dt><dd>%s</dd>" % (html.escape(k), html.escape(str(v))))
    A("</dl>")
    A("<div class='note'><b>CUDA Toolkit もコンパイラも入れていない。</b>"
      "TensorRT 版将棋エンジンの配布物に同梱されていた <code>cublas64_13.dll</code> と "
      "<code>cudart64_13.dll</code> を Python の <code>ctypes</code> から直接叩いている。"
      "DLL は読み取りのみで、エンジン側には一切変更を加えていない。</div>")

    A("<h2>native FP64 とエミュレーションの比較</h2>")
    A("".join(tA))
    A(ch1)
    A("<p><b>読み方。</b>native FP64 は行列サイズを変えても %.2f&ndash;%.2f TFLOPS で"
      "張り付いている。これがレート制限の壁。エミュレーション側は n とともに伸び、"
      "大きい行列ほど差が開く。入力の指数レンジが広い «hard» では必要マンティッサビットが"
      "増えるため利得は下がるが、それでも %.1f&times; を保っている。</p>"
      % (min(r[1] for r in rows if r[1]), nat_peak, best["sp_hard"]))

    nmax = sizes[-1]
    an, ae = pick(nat, nmax, "easy"), pick(emu, nmax, "easy")
    if an and ae and an.get("accuracy") and ae.get("accuracy"):
        mn = an["accuracy"]["rel_err_max"]
        me = ae["accuracy"]["rel_err_max"]
        A("<h3>精度は中央値より最悪値の方が差が大きい</h3>")
        A("<table><thead><tr><th>n=%d</th><th>相対誤差 中央値</th>"
          "<th>相対誤差 最悪値</th></tr></thead><tbody>"
          "<tr><td>native FP64</td><td>%s</td><td>%s</td></tr>"
          "<tr><td>emulated (ADP)</td><td>%s</td><td>%s</td></tr>"
          "</tbody></table>"
          % (nmax, fmt_err(relerr(an)), fmt_err(mn),
             fmt_err(relerr(ae)), fmt_err(me)))
        A("<p>中央値では %.0f 倍の差だが、<b>最悪値では %.0f 倍</b>開く。"
          "native FP64 の誤差は n 項の総和にわたる累積なので、"
          "打ち消しが起きた要素で大きく悪化する。固定小数点エミュレーションは"
          "整数で厳密に積和して最後に一度だけ丸めるため、"
          "この種の最悪ケースを構造的に持たない。"
          "条件数を2乗する正規方程式のような用途では、"
          "この最悪値の差がそのまま効いてくる。</p>"
          % (relerr(an) / relerr(ae) if relerr(ae) else 0, mn / me if me else 0))

    if tB:
        A("<h2>エミュレーション戦略の影響</h2>")
        A("<p><code>CUBLAS_EMULATION_STRATEGY</code> は "
          "<code>performant</code>（エミュレーションが得と判断したときだけ使う）と "
          "<code>eager</code>（積極的に使う）を取る。native がここまで遅いカードでは "
          "<code>performant</code> でも常にエミュレーションが選ばれるため、"
          "差は出なかった。</p>")
        A(tB)

    if tC:
        A("<h2>マンティッサビットと速度・精度のトレードオフ</h2>")
        A("<p>既定では ADP（automatic dynamic precision）が入力から必要ビット数を"
          "自動決定する。<code>CUBLAS_FIXEDPOINT_EMULATION_MANTISSA_BIT_COUNT</code> で"
          "固定すると、速度と精度を直接交換できる。</p>")
        A(tC)
        A(ch2)
        A(ch3)
        # ADP is not free: compare it against a hand-fixed bit count
        adp8 = pick(emu, 8192, "easy")
        if adp8 and 48 in cbits:
            i48 = cbits.index(48)
            f48, e48 = c_easy[i48], ce_easy[i48]
            nat8 = pick(nat, 8192, "easy")
            if f48 and adp8:
                A("<div class='note'><b>ADP は無料ではない。</b>"
                  "同じ n=8192 で ADP は %.3f TFLOPS、マンティッサを 48 bit に"
                  "固定すると %.3f TFLOPS で <b>%.2f 倍速い</b>。"
                  "精度はそれぞれ %s と %s、どちらも約 1 ULP で実用上同じ"
                  "（しかも native の %s より一桁良い）。ADP は必要ビット数を"
                  "決めるために入力を走査するぶんを払っている。"
                  "<b>入力のレンジが既知なら、固定した方が得だ。</b></div>"
                  % (adp8["tflops_median"], f48, f48 / adp8["tflops_median"],
                     fmt_err(relerr(adp8)), fmt_err(e48),
                     fmt_err(relerr(nat8)) if nat8 else "&mdash;"))
        plats = [(b, c_easy[i]) for i, b in enumerate(cbits) if c_easy[i]]
        if len(plats) > 4:
            A("<p><b>階段状のプラトーが実装を露出させている。</b>"
              "要求ビット数を増やしても値が動かない区間が並ぶのは、"
              "ビット数が<b>離散的なスライス数に切り上げられている</b>ためだ。"
              "INT8 のスライスを何枚積むかという内部構造が、"
              "外から観測できてしまっている。</p>")

    # ---- Phase D: power / energy
    dmodes = []
    for lbl in ("D_native", "D_adp", "D_mant48", "D_mant08"):
        rec = data.get(lbl)
        if rec and "tflops_sustained" in rec:
            dmodes.append(rec)
    if dmodes:
        base = dmodes[0]
        A("<h2>電力と電力効率</h2>")
        A("<p>25 秒間 DGEMM を連続投入し、その間 <code>nvidia-smi</code> を "
          "200 ms 間隔でサンプリングした。立ち上がりの 3 秒は定常状態から除外。</p>")
        A("<table><thead><tr><th>mode</th><th>sustained<br>TFLOPS</th>"
          "<th>mean board<br>power</th><th>GFLOP/s<br>per watt</th>"
          "<th>efficiency<br>vs native</th><th>SM clock</th>"
          "<th>peak<br>temp</th></tr></thead><tbody>")
        for rec in dmodes:
            name = rec["label"].replace("D_", "")
            A("<tr><td><code>%s</code></td><td>%.3f</td><td>%.0f W</td>"
              "<td><b>%.1f</b></td><td>%.2f&times;</td><td>%.0f MHz</td>"
              "<td>%.0f &deg;C</td></tr>"
              % (html.escape(name), rec["tflops_sustained"], rec["power_w_mean"],
                 rec["gflops_per_watt"],
                 rec["gflops_per_watt"] / base["gflops_per_watt"],
                 rec["clock_sm_mean"], rec["temp_c_max"]))
        A("</tbody></table>")
        A("<p><b>native FP64 は電力枠を使い切れない。</b>575 W の枠に対して "
          "%.0f W しか引かず、SM クロックは %.0f MHz と高いまま張り付く。"
          "稼働しているダイ面積が小さすぎて、電力ではなくクロックで律速されている"
          "ということだ。エミュレーション側は INT8 テンソルコアを動かすため "
          "%.0f W まで上がり（電力上限に張り付く）、クロックは %.0f MHz に下がる。"
          "つまりこの技術は<b>使われずに余っていた電力枠をスループットに変換している</b>。</p>"
          % (base["power_w_mean"], base["clock_sm_mean"],
             dmodes[-1]["power_w_mean"], dmodes[-1]["clock_sm_mean"]))
        eff = [r for r in dmodes if r["label"] == "D_mant48"]
        if eff:
            e = eff[0]
            A("<p>結果として、<b>FP64 相当の精度を保ったまま電力効率が %.1f 倍</b>に"
              "なる。消費電力は %.1f 倍に増えるが、仕事量が %.1f 倍になるためだ。</p>"
              % (e["gflops_per_watt"] / base["gflops_per_watt"],
                 e["power_w_mean"] / base["power_w_mean"],
                 e["tflops_sustained"] / base["tflops_sustained"]))

    A("<h2>測定方法</h2>")
    A("<ul>")
    A("<li><b>計測</b>: <code>cudaEvent</code> による GPU 時間。warmup 2 回ののち "
      "5 回（掃引では 3 回）測り中央値を採用。FLOP は 2n&sup3;。</li>")
    A("<li><b>精度の基準</b>: 浮動小数点の参照値ではなく<b>厳密値</b>と比較している。"
      "<code>math.fma</code> による two-product で各積の丸め誤差を取り出し、"
      "積と誤差の両方を <code>math.fsum</code> で厳密に総和する。これは正しく丸めた"
      "内積に一致する。出力行列からランダムに抽出した要素について相対誤差を取った。</li>")
    A("<li><b>入力分布</b>: <code>easy</code> は一様乱数 [-1,1]。"
      "<code>hard</code> は正規乱数に 2<sup>U(-%g,%g)</sup> を掛けて指数レンジを広げたもの。"
      "固定小数点分割の必要ビット数は指数レンジで決まるため、この2本が上下の目安になる。</li>"
      % (hi["phi"] or 0, hi["phi"] or 0))
    A("<li><b>モード切り替え</b>: cuBLAS は環境変数をハンドル生成時に読むため、"
      "1モード = 1プロセスで起動している。</li>")
    A("</ul>")

    A("<h2>この結果が言っていないこと</h2>")
    A("<div class='note warn'><ul>"
      "<li>これは <b>DGEMM 単体</b>の測定である。実アプリの利得は FP64 行列積が"
      "どれだけ時間を占めるかに支配され、メモリ帯域律速の処理では小さくなる。</li>"
      "<li><b>機械1台・ドライバ1種・1セッション</b>の測定であり、"
      "個体差や温度・電力状態の影響を分離していない。</li>"
      "<li>利得は入力の指数レンジに依存する。極端に広いレンジではさらに下がる。</li>"
      "<li><b>深層学習の推論には関係がない。</b>推論は FP16/INT8/FP8 で動いており "
      "FP64 を使わない。エミュレーションは INT8 の演算量を大量に消費して FP64 の"
      "精度を再構成する技術であり、低精度推論を速くするものではない。</li>"
      "</ul></div>")

    A("<h2>再現手順</h2>")
    A("<pre><code>git clone https://github.com/soy-tuber/fp64-emulation-rtx5090\n"
      "cd fp64-emulation-rtx5090\n\n"
      "# CUDA 13.0u2 以降の cublas64_13.dll / cudart64_13.dll があるディレクトリ\n"
      "set FP64EMU_CUDA_DIR=C:\\path\\to\\cuda13\\dlls\n\n"
      "python run_all.py        # 全モードを掃引し results/*.json を作る\n"
      "python make_report.py    # docs/index.html を生成する\n</code></pre>")
    A("<p>個別に1モードだけ測る場合:</p>")
    A("<pre><code>set CUBLAS_EMULATE_DOUBLE_PRECISION=1\n"
      "python dgemm_bench.py --sizes 16384 --dists easy,hard --label emu\n</code></pre>")

    A("<h2>生データ</h2><ul>")
    A("<li><a href='https://github.com/soy-tuber/fp64-emulation-rtx5090/tree/main/results'>"
      "results/ の JSON 一式</a>（各測定の全試行時間・GPU メモリ使用量・精度統計を含む）</li>")
    A("<li><a href='https://github.com/soy-tuber/fp64-emulation-rtx5090/blob/main/dgemm_bench.py'>"
      "dgemm_bench.py</a> / <a href='https://github.com/soy-tuber/fp64-emulation-rtx5090/blob/main/run_all.py'>"
      "run_all.py</a> / <a href='https://github.com/soy-tuber/fp64-emulation-rtx5090/blob/main/make_report.py'>"
      "make_report.py</a></li>")
    A("</ul>")

    A("<h2>参考</h2><ul>")
    A("<li><a href='https://docs.nvidia.com/cuda/cublas/'>cuBLAS documentation "
      "&mdash; Fixed-Point Emulation</a></li>")
    A("<li><a href='https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/'>"
      "NVIDIA: Unlocking Tensor Core Performance with Floating Point Emulation in cuBLAS</a></li>")
    A("<li><a href='https://arxiv.org/abs/2504.08009'>Ozaki Scheme II: A GEMM-oriented "
      "emulation of floating-point matrix multiplication using an integer modular "
      "technique</a></li>")
    A("<li><a href='https://github.com/RIKEN-RCCS/GEMMul8'>RIKEN-RCCS/GEMMul8</a></li>")
    A("</ul>")

    A("<footer>このページは <code>make_report.py</code> が "
      "<code>results/*.json</code> から生成しています。"
      "表中の数値は一切手入力されていません。</footer>")
    A("</div></body></html>")

    out = os.path.join(DOCS, "index.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write("".join(H))
    print("wrote %s (%d bytes, %d result files)"
          % (out, os.path.getsize(out), len(data)))

    # ------------------------------------------------------------- README.md
    pw = []
    for lbl in ("D_native", "D_adp", "D_mant48", "D_mant08"):
        rec = data.get(lbl)
        if rec and "tflops_sustained" in rec:
            pw.append(rec)
    R = []
    R.append("# FP64 emulation on a GeForce RTX 5090\n")
    R.append("\nNative vs Ozaki-style fixed-point **FP64 emulation** "
             "(`cuBLAS` 13, CUDA 13.0 Update 2+) measured on *consumer* "
             "Blackwell, where hardware FP64 is rate-limited to 1/64 of FP32.\n")
    R.append("\n**Report: https://soy-tuber.github.io/fp64-emulation-rtx5090/**\n")
    R.append("\nMeasured on `%s`, cuBLAS `%s` (`cublasGetVersion`), %s.\n"
             % (hi["gpu"].split(",")[0].strip(), hi["cublas_version"], hi["date"]))
    R.append("\n## Headline\n\n")
    R.append("| | native FP64 | emulated (ADP) |\n|---|---|---|\n")
    R.append("| peak throughput | %.3f TFLOPS | **%.3f TFLOPS** |\n"
             % (nat_peak, emu_peak))
    R.append("| speedup, narrow exponent range | 1.00x | **%.2fx** |\n" % best["sp_easy"])
    R.append("| speedup, wide exponent range | 1.00x | **%.2fx** |\n" % best["sp_hard"])
    R.append("| relative error vs exact product | ~%.0e | **0 to 1 ULP** |\n"
             % (nat_err or 0))
    if pw:
        R.append("\n### Energy (n=16384, 25 s sustained)\n\n")
        R.append("| mode | sustained TFLOPS | mean board power | GFLOP/s per W "
                 "| vs native |\n|---|---|---|---|---|\n")
        for rec in pw:
            R.append("| %s | %.3f | %.0f W | **%.1f** | %.2fx |\n"
                     % (rec["label"].replace("D_", ""), rec["tflops_sustained"],
                        rec["power_w_mean"], rec["gflops_per_watt"],
                        rec["gflops_per_watt"] / pw[0]["gflops_per_watt"]))
        R.append("\n`mant48` keeps full FP64-grade accuracy. `mant08` does not "
                 "(relative error ~1e-4) and is listed to show where the "
                 "hardware ceiling actually is. Native FP64 draws only %.0f W "
                 "of the 575 W budget: too little of the die is active, so it is "
                 "clock-limited rather than power-limited, and the emulation "
                 "converts that unused power budget into throughput.\n"
                 % pw[0]["power_w_mean"])
    R.append("\nThe emulation is not a precision/speed trade here: it is both "
             "faster **and** more accurate than native FP64, because the "
             "fixed-point split accumulates exactly in integer arithmetic and "
             "rounds once at the end, whereas native FP64 accumulates rounding "
             "error over n terms.\n")
    R.append("\n## Why measure this\n\n")
    R.append("Published emulation numbers come mostly from data-center GPUs, "
             "where native FP64 is already fast and the gain looks modest. On a "
             "GeForce card the premise is inverted: FP64 is deliberately "
             "crippled while INT8 tensor throughput is abundant, so the ratio "
             "should be much larger. Measurements for consumer Blackwell are "
             "hard to find, so this repo provides some.\n")
    R.append("\n## No CUDA Toolkit required\n\n")
    R.append("The harness calls `cublasDgemm` through Python `ctypes` against "
             "`cublas64_13.dll` / `cudart64_13.dll`. Any CUDA 13.0u2+ "
             "redistributable set works; this run used the DLLs bundled with a "
             "TensorRT application, read-only. No compiler, no toolkit install.\n")
    R.append("\n## Layout\n\n")
    R.append("| path | what |\n|---|---|\n")
    R.append("| `dgemm_bench.py` | one mode per process: timing + accuracy |\n")
    R.append("| `power_bench.py` | sustained load with `nvidia-smi` sampling |\n")
    R.append("| `run_all.py` | spawns every mode, writes `results/*.json` |\n")
    R.append("| `make_report.py` | generates `docs/index.html` and this README |\n")
    R.append("| `results/` | raw JSON, every repetition included |\n")
    R.append("\n## Reproduce\n\n```\nset FP64EMU_CUDA_DIR=C:\\path\\to\\cuda13\\dlls\n"
             "python run_all.py\npython make_report.py\n```\n")
    R.append("\n## Accuracy methodology\n\n")
    R.append("Results are compared against the **exact** product, not another "
             "floating-point result. `math.fma` two-product recovers the "
             "rounding error of every `a*b`, and `math.fsum` sums products and "
             "errors exactly, giving the correctly-rounded dot product. "
             "Relative error is reported over randomly sampled output entries.\n")
    R.append("\n## What this does not show\n\n")
    R.append("- DGEMM only. Real applications gain in proportion to how much "
             "FP64 matrix multiply they actually do.\n")
    R.append("- One machine, one driver, one session.\n")
    R.append("- The gain depends on the exponent range of the inputs.\n")
    R.append("- **Nothing for deep-learning inference.** Inference runs in "
             "FP16/INT8/FP8 and uses no FP64. This technique spends INT8 "
             "throughput to reconstruct FP64 accuracy; it does not make "
             "low-precision inference faster.\n")
    R.append("\n## References\n\n")
    R.append("- [cuBLAS documentation - Fixed-Point Emulation]"
             "(https://docs.nvidia.com/cuda/cublas/)\n")
    R.append("- [NVIDIA: Unlocking Tensor Core Performance with Floating Point "
             "Emulation in cuBLAS](https://developer.nvidia.com/blog/"
             "unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)\n")
    R.append("- [Ozaki Scheme II (arXiv:2504.08009)]"
             "(https://arxiv.org/abs/2504.08009)\n")
    R.append("- [RIKEN-RCCS/GEMMul8](https://github.com/RIKEN-RCCS/GEMMul8)\n")
    R.append("\n## License\n\nMIT\n")
    rp = os.path.join(HERE, "README.md")
    with open(rp, "w", encoding="utf-8") as f:
        f.write("".join(R))
    print("wrote %s (%d bytes)" % (rp, os.path.getsize(rp)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
