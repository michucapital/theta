# gen_replay_eqsignal.py
# ============================================================================
# CONFIGURATION
# ============================================================================
ChooseDay       = "20260513"
RandomDays      = False
CUM_WARMUP      = 30_000
STRIDE          = 5
STRIDE_VOL      = 50
CHUNK_TICKS     = 1_000
CHUNK_LARGE     = 10_000
Y_INIT_RANGE    = 3.0
Y_MARGIN        = 0.2
PT125_LINE      = 40_000.0
PRICE_DECIMALS  = 2

# ── Lower subplot Y-axis ────────────────────────────────────────────────────
EQSIGNAL_Y_INIT   = [-0.5, 0.5]    # initial [lo, hi] — range = 1.0
EQSIGNAL_Y_MARGIN = 0.1            # dynamic expansion margin
# ============================================================================

from pathlib import Path
import pandas as pd
import numpy as np
import json, random, traceback, base64, struct
from tqdm import tqdm

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        return super().default(obj)

def jdump(obj):
    return json.dumps(obj, cls=NumpyEncoder, separators=(',',':'))

def delta_encode(arr, decimals=PRICE_DECIMALS):
    scale = 10 ** decimals
    nulls, deltas = [], []
    prev = None
    for i, v in enumerate(arr):
        if v != v:
            nulls.append(i); deltas.append(0)
        else:
            r = round(float(v), decimals)
            if prev is None: prev = r
            deltas.append(round((r - prev) * scale))
            prev = r
    fi = next((i for i, v in enumerate(arr) if v == v), 0)
    fv = round(float(arr[fi]), decimals) if fi < len(arr) else 0.0
    return {"s": fv, "fi": int(fi), "sc": scale, "d": deltas, "n": nulls}

def pack_f32_b64(arr):
    return base64.b64encode(struct.pack(f"{len(arr)}f", *map(float, arr))).decode()

def get_xtick_info(time_np, stride):
    targets = [(34_200_000,"9:30"),(36_300_000,"10:05"),(39_600_000,"11:00"),
               (43_200_000,"12:00"),(46_800_000,"13:00"),(50_400_000,"14:00"),
               (53_100_000,"14:45")]
    tv, tt = [], []
    for ms, label in targets:
        idx = int(np.argmin(np.abs(time_np - ms)))
        if abs(time_np[idx] - ms) <= 1_800_000:
            tv.append(int(idx // stride)); tt.append(label)
    return tv, tt

def rng_id(used):
    while True:
        v = random.randint(1, 1000)
        if v not in used: used.add(v); return v

def compute_yranges(sp, cs, ps, eq12, eq25, eqitm, total_pts, chunk_pts):
    valid_sp = sp[~np.isnan(sp)]
    first_price = float(valid_sp[0]) if len(valid_sp) else 0.0
    cur_lo = first_price - Y_INIT_RANGE / 2
    cur_hi = first_price + Y_INIT_RANGE / 2
    boundaries = list(range(0, total_pts + 1, chunk_pts))
    if boundaries[-1] != total_pts: boundaries.append(total_pts)
    ranges = []
    for end_idx in boundaries:
        if end_idx > 0:
            slices = [a[:end_idx][~np.isnan(a[:end_idx])] for a in (sp, cs, ps, eq12, eq25, eqitm)]
            slices = [s for s in slices if len(s)]
            if slices:
                all_v  = np.concatenate(slices)
                cur_lo = min(cur_lo, float(np.min(all_v)) - Y_MARGIN)
                cur_hi = max(cur_hi, float(np.max(all_v)) + Y_MARGIN)
                span   = cur_hi - cur_lo
                if span < Y_INIT_RANGE:
                    pad = (Y_INIT_RANGE - span) / 2
                    cur_lo -= pad; cur_hi += pad
        ranges.append([round(cur_lo, 4), round(cur_hi, 4)])
    return ranges

def compute_eqsignal_yranges(eqsignal_s, total_pts, chunk_pts):
    cur_lo, cur_hi = list(EQSIGNAL_Y_INIT)
    boundaries = list(range(0, total_pts + 1, chunk_pts))
    if boundaries[-1] != total_pts: boundaries.append(total_pts)
    ranges = []
    for end_idx in boundaries:
        if end_idx > 0:
            sl = eqsignal_s[:end_idx]
            valid = sl[~np.isnan(sl)]
            if len(valid):
                cur_lo = min(cur_lo, float(np.min(valid)) - EQSIGNAL_Y_MARGIN)
                cur_hi = max(cur_hi, float(np.max(valid)) + EQSIGNAL_Y_MARGIN)
        ranges.append([round(cur_lo, 4), round(cur_hi, 4)])
    return ranges

def load_vix_gap_db(script_dir: Path) -> dict:
    db_path = script_dir / "data" / "vix_gap" / "vix_gap_db.csv"
    if not db_path.exists(): return {}
    df = pd.read_csv(db_path, dtype={"DATE": str})
    df.columns = [c.strip().upper() for c in df.columns]
    return {str(r["DATE"]).strip().zfill(8): (float(r["VIX_OPEN"]), float(r["GAP"]))
            for _, r in df.iterrows()}

HTML_TEMPLATE = """\\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>SPY Replay (EqSignal & EqITM)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* {{box-sizing:border-box;margin:0;padding:0}}
body {{background:#1a1a1a;display:flex;flex-direction:column;
       align-items:center;font-family:sans-serif;height:100vh;overflow:hidden}}
#chart {{width:100vw;flex:1;min-height:0;position:relative}}
#crosshair-h {{position:absolute;left:0;right:0;height:1px;
               background:rgba(200,200,200,0.4);pointer-events:none;display:none}}
#controls {{display:flex;gap:14px;padding:10px 16px;background:#111;
            width:100%;justify-content:center;align-items:center;flex-shrink:0}}
button {{padding:10px 22px;font-size:15px;font-weight:bold;cursor:pointer;
         border:none;border-radius:6px;transition:opacity .15s}}
button:hover {{opacity:.82}}
#btn_bl     {{background:#c0392b;color:#fff}}
#btn_bs     {{background:#e74c3c;color:#fff}}
#btn_fs     {{background:#27ae60;color:#fff}}
#btn_fl     {{background:#1e8449;color:#fff}}
#btn_reveal {{background:#4a4a8a;color:#fff}}
#lbl    {{color:#aaa;font-size:15px;min-width:110px;text-align:center}}
#vixgap {{color:#fff;font-size:15px;font-weight:bold;
          margin-right:auto;padding-left:8px;white-space:nowrap}}
#spacer {{margin-left:auto}}
</style>
</head>
<body>
<div id="chart"><div id="crosshair-h"></div></div>
<div id="controls">
  <span id="vixgap">{VIX_GAP_TEXT}</span>
  <button id="btn_bl" onclick="step(-{CHUNK_LARGE_PTS})">&#8722; 10k</button>
  <button id="btn_bs" onclick="step(-{CHUNK_SMALL_PTS})">&#8722; 1k</button>
  <span id="lbl">chunk 1 / {TOTAL_CHUNKS}</span>
  <button id="btn_fs" onclick="step(+{CHUNK_SMALL_PTS})">&#43; 1k</button>
  <button id="btn_fl" onclick="step(+{CHUNK_LARGE_PTS})">&#43; 10k</button>
  <button id="btn_reveal" onclick="reveal()">Reveal</button>
  <span id="spacer"></span>
</div>
<script>
const TOTAL_PTS    = {TOTAL_PTS};
const CHUNK_SMALL  = {CHUNK_SMALL_PTS};
const CHUNK_LARGE  = {CHUNK_LARGE_PTS};
const TOTAL_CHUNKS = {TOTAL_CHUNKS};
const Y_RANGES     = {Y_RANGES_JSON};
const TICK_VALS    = {TICKVALS_JSON};
const TICK_TEXT    = {TICKTEXT_JSON};
const PT125_REF    = {PT125_LINE};
const PT_MAX       = {PT_MAX};
const VOL_RATIO    = {VOL_RATIO};
const TIME_MS      = {TIME_MS_JSON};
const EQSIGNAL_RANGES = {EQSIGNAL_RANGES_JSON};

// MaxMin specific arrays
const MAXMIN_POS   = {MAXMIN_POS_JSON};
const MAXMIN_NEG   = {MAXMIN_NEG_JSON};

function decode(enc) {{
  const {{s,fi,sc,d,n}} = enc;
  const out = new Array(d.length).fill(null);
  const ns = new Set(n);
  let cur = s;
  for (let i=fi;i<d.length;i++) {{
    if (ns.has(i)) {{ out[i]=null; continue; }}
    cur += d[i]/sc; out[i] = Math.round(cur*sc)/sc;
  }}
  return out;
}}
function decodeF32(b64) {{
  const bin=atob(b64),buf=new ArrayBuffer(bin.length),u8=new Uint8Array(buf);
  for (let i=0;i<bin.length;i++) u8[i]=bin.charCodeAt(i);
  return Array.from(new Float32Array(buf));
}}

const raw_sp       = decode({SP_ENC});
const raw_eq12     = decode({EQ12_ENC});
const raw_cs       = decode({CS_ENC});
const raw_ps       = decode({PS_ENC});
const raw_pt       = decodeF32("{PT_B64}");
const raw_eq25     = decode({EQ25_ENC});
const raw_eqsignal = decode({EQSIGNAL_ENC});
const raw_eqitm    = decode({EQITM_ENC});

let curPts = CHUNK_SMALL;

function msToTime(ms) {{
  if (ms==null) return "";
  const m=Math.round(ms/60000), h=Math.floor(m/60);
  return h+":"+String(m%60).padStart(2,"0");
}}

// ── Main chart ¼-dollar gridlines ─────────────────────────────────────────────
function quarterShapes(yr) {{
  const shapes=[];
  for (let v=Math.ceil(yr[0]*4)/4; v<=yr[1]+0.001; v+=0.25) {{
    const rv=Math.round(v*4)/4;
    if (Math.abs(rv-Math.round(rv))<0.01) continue;
    shapes.push({{type:"line",xref:"paper",yref:"y",x0:0,x1:1,y0:rv,y1:rv,
      line:{{color:"rgba(80,80,80,0.6)",width:0.5}},layer:"below"}});
  }}
  shapes.push({{type:"line",xref:"paper",yref:"y3",x0:0,x1:1,y0:0,y1:0,
    line:{{color:"rgba(220,50,50,0.85)",width:1.2}},layer:"below"}});
  return shapes;
}}

// ── Get All Shapes (Gridlines + MaxMin Vertical Lines) ────────────────────────
function getShapes(pts, yr) {{
  const shapes = quarterShapes(yr);
  
  for (let i=0; i<MAXMIN_POS.length; i++) {{
    const xIdx = MAXMIN_POS[i];
    if (xIdx >= pts) break;
    shapes.push({{
      type: "line", xref: "x", yref: "y3 domain", 
      x0: xIdx, x1: xIdx, y0: 0, y1: 1,
      line: {{color: "rgba(0,255,0,0.7)", width: 1.5, dash: "dot"}}, layer: "below"
    }});
  }}
  
  for (let i=0; i<MAXMIN_NEG.length; i++) {{
    const xIdx = MAXMIN_NEG[i];
    if (xIdx >= pts) break;
    shapes.push({{
      type: "line", xref: "x", yref: "y3 domain", 
      x0: xIdx, x1: xIdx, y0: 0, y1: 1,
      line: {{color: "rgba(255,0,0,0.7)", width: 1.5, dash: "dot"}}, layer: "below"
    }});
  }}
  
  return shapes;
}}

function chunkIdx(pts) {{
  return Math.min(Math.round(pts/CHUNK_SMALL),Y_RANGES.length-1);
}}

function buildLayout(pts) {{
  const yr = Y_RANGES[chunkIdx(pts)];
  return {{
    paper_bgcolor:"#1a1a1a", plot_bgcolor:"#1e1e1e", font:{{color:"#cccccc"}},
    margin:{{t:20, b:40, l:60, r:60}},

    xaxis:{{
      range:[0,TOTAL_PTS-1], tickmode:"array", tickvals:TICK_VALS, ticktext:TICK_TEXT,
      showgrid:true, gridcolor:"#333", fixedrange:true, anchor:"y3",
    }},

    // ── Main price chart ──────────────────────────────────────────────────────
    yaxis:{{
      domain:[0.32, 1.0], range:yr, showgrid:true, gridcolor:"#444",
      dtick:1, fixedrange:true,
    }},
    yaxis2:{{
      overlaying:"y", side:"right", showgrid:false, showticklabels:false,
      range:[0, PT_MAX*10], fixedrange:true,
    }},

    // ── Lower subplot (EqSignal ONLY) ─────────────────────────────────────────
    yaxis3:{{
      domain:[0.0, 0.27], range:EQSIGNAL_RANGES[chunkIdx(pts)],
      showgrid:true, gridcolor:"#444",
      dtick:0.5, tickformat:".2f", fixedrange:true, anchor:"x",
    }},

    shapes: getShapes(pts, yr),
    legend:{{orientation:"h", y:-0.06, x:0.5, xanchor:"center",
             bgcolor:"rgba(0,0,0,0)", font:{{size:12}}}},
    hovermode:"x unified",
  }};
}}

// ── SPY line — always white ───────────────────────────────────────────────────
function spyColorTraces(pts, sp) {{
  const xs=Array.from({{length:pts}},(_,i)=>i);
  return [{{x:xs,y:sp.slice(0,pts),mode:"lines",type:"scatter",
    name:"SPY Price",showlegend:true,connectgaps:false,
    line:{{width:1.8,color:"#ffffff"}},
    hovertemplate:"<b>%{{customdata}}</b><br>SPY&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>",
    customdata:xs.map(i=>msToTime(TIME_MS[i])),
  }}];
}}

// ── CallStrike trace ──────────────────────────────────────────────────────────
function callStrikeTraces(pts, cs) {{
  const xs=Array.from({{length:pts}},(_,i)=>i);
  return [{{x:xs,y:cs.slice(0,pts),mode:"lines",type:"scatter",
    name:"CallStrike",showlegend:true,connectgaps:false,
    line:{{width:1.4,color:"rgba(0,210,90,0.90)",dash:"dash"}},
    hovertemplate:"CallStrike&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>"
  }}];
}}

function maskedTraces(pts) {{
  const xs=Array.from({{length:pts}},(_,i)=>i);
  const eq12=raw_eq12.slice(0,pts);
  const eq25=raw_eq25.slice(0,pts);
  const eqitm=raw_eqitm.slice(0,pts);
  const cs=raw_cs.slice(0,pts), ps=raw_ps.slice(0,pts), sp=raw_sp.slice(0,pts);
  const vPts=Math.ceil(pts/VOL_RATIO);
  const vxs=Array.from({{length:vPts}},(_,i)=>i*VOL_RATIO);
  const pt=raw_pt.slice(0,vPts);
  const eqsignal = raw_eqsignal.slice(0,pts);

  return [
    ...spyColorTraces(pts, sp),
    {{x:xs,y:eq12,name:"Eq12",mode:"lines",connectgaps:false,
      line:{{width:1.2,color:"rgba(255,220,0,0.90)"}},
      hovertemplate:"Eq12&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>"}},
    {{x:xs,y:eq25,name:"Eq25",mode:"lines",connectgaps:false,
      line:{{width:1.2,color:"rgba(30,144,255,0.90)"}},
      hovertemplate:"Eq25&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>"}},
    // ── EqITM added to main upper chart ───────────────────────────────────────
    {{x:xs,y:eqitm,name:"EqITM",mode:"lines",connectgaps:false,
      line:{{width:1.2,color:"rgba(200,150,255,0.90)"}}, // Light Purple
      hovertemplate:"EqITM&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>"}},
    ...callStrikeTraces(pts, cs),
    {{x:xs,y:ps,name:"PutStrike",mode:"lines",connectgaps:false,
      line:{{width:1.4,color:"rgba(230,60,60,0.90)",dash:"dash"}},
      hovertemplate:"PutStrike&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.2f}}<extra></extra>"}},
    {{x:vxs,y:pt,name:"Volume",type:"bar",yaxis:"y2",
      marker:{{color:"rgba(100,160,255,0.45)",line:{{width:0}}}},
      hovertemplate:"Volume&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;%{{y:,.0f}}<extra></extra>",showlegend:true}},
    {{x:vxs,y:vxs.map(()=>PT125_REF),yaxis:"y2",mode:"lines",
      hoverinfo:"skip",showlegend:false,
      line:{{color:"rgba(220,60,60,0.65)",width:0.8,dash:"dot"}}}},
    // ── EqSignal ONLY on lower subplot ────────────────────────────────────────
    {{x:xs,y:eqsignal,yaxis:"y3",mode:"lines",type:"scatter",
      name:"EqSignal",showlegend:true,connectgaps:false,
      line:{{width:1.5,color:"rgba(255,165,0,0.90)"}},
      hovertemplate:"EqSignal&nbsp;&nbsp;&nbsp;&nbsp;%{{y:.4f}}<extra></extra>"}},
  ];
}}

Plotly.newPlot("chart",maskedTraces(curPts),buildLayout(curPts),
  {{displayModeBar:false,responsive:true}});

const chartDiv=document.getElementById("chart");
const hLine=document.getElementById("crosshair-h");
chartDiv.addEventListener("mousemove",e=>{{
  hLine.style.top=(e.clientY-chartDiv.getBoundingClientRect().top)+"px";
  hLine.style.display="block";
}});
chartDiv.addEventListener("mouseleave",()=>hLine.style.display="none");

function step(delta) {{
  curPts=Math.max(CHUNK_SMALL,Math.min(TOTAL_PTS,curPts+delta));
  Plotly.react("chart",maskedTraces(curPts),buildLayout(curPts),
    {{displayModeBar:false,responsive:true}});
  document.getElementById("lbl").textContent=
    "chunk "+Math.round(curPts/CHUNK_SMALL)+" / "+TOTAL_CHUNKS;
}}

function reveal() {{
  curPts=TOTAL_PTS;
  Plotly.react("chart",maskedTraces(curPts),buildLayout(curPts),
    {{displayModeBar:false,responsive:true}});
  document.getElementById("lbl").textContent="chunk "+TOTAL_CHUNKS+" / "+TOTAL_CHUNKS;
}}
</script>
</body>
</html>
"""

def process_file(file_path, out_folder, vix_gap_lookup, used_ids):
    df = pd.read_parquet(file_path)
    date_str = file_path.stem.split("_")[1]

    req = ["TIME","spot_price","CallStrike","PutStrike","Eq12","Eq25", "EqSignal", "EqITM", "MaxMin"]
    missing = [c for c in req if c not in df.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")

    has_pt     = "Premium_Traded125" in df.columns

    time_np = df["TIME"].values.astype(np.float64)
    sp_raw  = df["spot_price"].values.astype(np.float64)
    cs_raw  = df["CallStrike"].values.astype(np.float64)
    ps_raw  = df["PutStrike"].values.astype(np.float64)
    eq12_raw  = df["Eq12"].values.astype(np.float64)
    pt_raw  = df["Premium_Traded125"].values.astype(np.float64) if has_pt else np.zeros(len(sp_raw))
    eq25_raw     = df["Eq25"].values.astype(np.float64)
    eqsignal_raw = df["EqSignal"].values.astype(np.float64)
    eqitm_raw    = df["EqITM"].values.astype(np.float64)
    maxmin_raw   = df["MaxMin"].values.astype(np.int8)

    pt = np.where(np.isnan(pt_raw), 0.0, pt_raw)

    sp_s          = sp_raw[::STRIDE]
    eq12_s        = eq12_raw[::STRIDE].copy();  eq12_s[:CUM_WARMUP//STRIDE] = np.nan
    cs_s          = np.where(cs_raw[::STRIDE]==0, np.nan, cs_raw[::STRIDE].copy())
    ps_s          = np.where(ps_raw[::STRIDE]==0, np.nan, ps_raw[::STRIDE].copy())
    cs_s[:CUM_WARMUP//STRIDE] = np.nan
    ps_s[:CUM_WARMUP//STRIDE] = np.nan
    eq25_s        = eq25_raw[::STRIDE].copy(); eq25_s[:CUM_WARMUP//STRIDE] = np.nan
    eqsignal_s    = eqsignal_raw[::STRIDE]
    eqitm_s       = eqitm_raw[::STRIDE].copy(); eqitm_s[:CUM_WARMUP//STRIDE] = np.nan
    pt_s          = pt[::STRIDE_VOL]
    time_s        = time_np[::STRIDE]

    # Map the original MaxMin ticks to our downsampled X-axis
    mm_pos_raw = np.where(maxmin_raw == 1)[0]
    mm_neg_raw = np.where(maxmin_raw == -1)[0]
    mm_pos_s   = sorted(list(set((mm_pos_raw // STRIDE).tolist())))
    mm_neg_s   = sorted(list(set((mm_neg_raw // STRIDE).tolist())))

    total_pts    = len(sp_s)
    chunk_s_pts  = max(1, CHUNK_TICKS  // STRIDE)
    chunk_l_pts  = max(1, CHUNK_LARGE  // STRIDE)
    total_chunks = (total_pts + chunk_s_pts - 1) // chunk_s_pts
    vol_ratio    = STRIDE_VOL // STRIDE

    tickvals, ticktext = get_xtick_info(time_np, STRIDE)
    # Include eqitm_s in the main chart's Y-range calculation
    y_ranges        = compute_yranges(sp_s, cs_s, ps_s, eq12_s, eq25_s, eqitm_s, total_pts, chunk_s_pts)
    eqsignal_ranges = compute_eqsignal_yranges(eqsignal_s, total_pts, chunk_s_pts)

    pt_valid = pt_s[pt_s > 0]
    pt_max   = float(np.percentile(pt_valid, 99)) if len(pt_valid) else float(PT125_LINE)

    if date_str in vix_gap_lookup:
        vix_open, gap = vix_gap_lookup[date_str]
        vix_gap_text  = f"VIX = {vix_open:.2f}   GAP = {gap:+.2f}"
    else:
        vix_gap_text = ""
        tqdm.write(f"  WARN: {date_str} not found in vix_gap_db.csv")

    rid      = rng_id(used_ids)
    out_path = out_folder / f"spy_{rid}_eqsignal.html"

    out_path.write_text(HTML_TEMPLATE.format(
        CHUNK_SMALL_PTS    = chunk_s_pts,
        CHUNK_LARGE_PTS    = chunk_l_pts,
        TOTAL_PTS          = total_pts,
        TOTAL_CHUNKS       = total_chunks,
        VOL_RATIO          = vol_ratio,
        Y_RANGES_JSON      = jdump(y_ranges),
        TICKVALS_JSON      = jdump(tickvals),
        TICKTEXT_JSON      = jdump(ticktext),
        TIME_MS_JSON       = jdump([int(v) for v in time_s]),
        SP_ENC             = jdump(delta_encode(sp_s)),
        EQ12_ENC           = jdump(delta_encode(eq12_s)),
        CS_ENC             = jdump(delta_encode(cs_s)),
        PS_ENC             = jdump(delta_encode(ps_s)),
        PT_B64             = pack_f32_b64(pt_s),
        PT125_LINE         = PT125_LINE,
        PT_MAX             = round(pt_max, 2),
        VIX_GAP_TEXT       = vix_gap_text,
        EQSIGNAL_RANGES_JSON = jdump(eqsignal_ranges),
        EQ25_ENC           = jdump(delta_encode(eq25_s)),
        EQSIGNAL_ENC       = jdump(delta_encode(eqsignal_s, decimals=4)),
        EQITM_ENC          = jdump(delta_encode(eqitm_s)), # Uses standard 2 decimals as a price level
        MAXMIN_POS_JSON    = jdump(mm_pos_s),
        MAXMIN_NEG_JSON    = jdump(mm_neg_s),
    ), encoding="utf-8")
    return True

def main():
    script_dir     = Path(__file__).resolve().parent
    data_folder    = script_dir / "data" / "merged"
    out_folder     = script_dir / "data" / "replay"
    out_folder.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_folder.glob("spy_*_merged.parquet"))
    if not all_files: print("ERROR: no parquet files found"); return

    selected = (select_random_days(all_files) if RandomDays
                else select_specific_day(all_files, ChooseDay))
    if not selected: print("ERROR: no files selected"); return

    vix_gap_lookup = load_vix_gap_db(script_dir)
    used_ids: set = set()
    ok = err = 0

    for fp in tqdm(selected, desc="Generating", unit="file", leave=True):
        try:
            if process_file(fp, out_folder, vix_gap_lookup, used_ids): ok += 1
            else: err += 1
        except Exception as e:
            tqdm.write(f"  ERROR: {e}"); tqdm.write(traceback.format_exc()); err += 1

    print(f"Done — {ok} file(s) saved to {out_folder}"
          + (f"  ({err} error(s))" if err else ""))

def select_specific_day(all_files, chosen_date):
    for fp in all_files:
        if fp.stem.split("_")[1] == chosen_date: return [fp]
    print("ERROR: requested day not found"); return []

def select_random_days(all_files):
    quarters = {
        "Q1_2025":("20250101","20250331"), "Q2_2025":("20250401","20250630"),
        "Q3_2025":("20250701","20250930"), "Q4_2025":("20251001","20251231"),
        "Q5_2026":("20260101","20260131"),
    }
    fbq = {q:[] for q in quarters}
    for f in all_files:
        d = f.stem.split("_")[1]
        for q,(s,e) in quarters.items():
            if s<=d<=e: fbq[q].append(f); break
    selected = [random.choice(v) for v in fbq.values() if v]
    needed = 10 - len(selected)
    if needed > 0:
        available = [f for f in all_files if f not in selected]
        sel_dates = {int(f.stem.split("_")[1]) for f in selected}
        pool = [f for f in available
                if (int(f.stem.split("_")[1])-1) not in sel_dates
                and (int(f.stem.split("_")[1])+1) not in sel_dates] or available
        selected.extend(random.sample(pool, min(needed, len(pool))))
    return sorted(selected)

if __name__ == "__main__":
    main()
