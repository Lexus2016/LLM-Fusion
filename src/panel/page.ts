/**
 * Self-contained connector + config panel (inline CSS + JS, no external requests).
 * Three tabs:
 *   - Monitor   : live provider/account health (polls /admin/providers).
 *   - Providers : no-YAML editor for provider groups + accounts.
 *   - Models    : no-YAML editor for virtual models (single/failover/fusion/smart).
 * All updates are applied IN PLACE on the monitor tab (keyed reconcile — no
 * flicker). Every mutating / destructive action asks for confirmation first. The
 * config editor never shows YAML; it POSTs structured objects the server
 * validates, backs up, and writes (comment-preserving).
 *
 * The client script deliberately avoids template literals — no backtick/`${` escaping.
 */
export const PANEL_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>llm-fusion · panel</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%235b9dff'/%3E%3C/svg%3E" />
<style>
  :root{
    --bg:#0b0f14; --panel:#121821; --panel-2:#0e141c; --line:#1e2733; --line-2:#273244;
    --fg:#e6edf3; --muted:#8b98a9; --muted-2:#5b6672;
    --up:#22c55e; --cooling:#f59e0b; --down:#ef4444; --off:#64748b; --accent:#5b9dff;
    --up-bg:rgba(34,197,94,.12); --cooling-bg:rgba(245,158,11,.12);
    --down-bg:rgba(239,68,68,.12); --off-bg:rgba(100,116,139,.12); --accent-bg:rgba(91,157,255,.12);
    --shadow:0 1px 2px rgba(0,0,0,.4),0 8px 24px rgba(0,0,0,.28);
    --radius:14px;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
    --sans:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  @media (prefers-color-scheme: light){
    :root{
      --bg:#f4f6f9; --panel:#ffffff; --panel-2:#f8fafc; --line:#e6eaf0; --line-2:#d7dde7;
      --fg:#101725; --muted:#5b6672; --muted-2:#8b98a9;
      --up-bg:rgba(34,197,94,.14); --cooling-bg:rgba(245,158,11,.16);
      --down-bg:rgba(239,68,68,.12); --off-bg:rgba(100,116,139,.14); --accent-bg:rgba(43,108,224,.10);
      --accent:#2b6ce0; --shadow:0 1px 2px rgba(16,23,37,.06),0 10px 26px rgba(16,23,37,.08);
    }
  }
  *{box-sizing:border-box}
  html,body{margin:0}
  body{background:var(--bg); color:var(--fg); font-family:var(--sans); font-size:15px; line-height:1.5;
    -webkit-font-smoothing:antialiased; padding:28px 22px 64px; max-width:1180px; margin:0 auto}
  .num{font-variant-numeric:tabular-nums}
  .mono{font-family:var(--mono)}
  .muted{color:var(--muted)}

  header.top{display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:16px}
  .brand{display:flex; align-items:center; gap:10px; font-weight:650; font-size:16px; letter-spacing:.2px}
  .brand .logo{width:22px;height:22px;border-radius:7px;background:linear-gradient(135deg,var(--accent),#9b7bff); box-shadow:var(--shadow)}
  .spacer{flex:1}
  .pill{display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px; font-size:12.5px; font-weight:600; border:1px solid var(--line-2); background:var(--panel)}
  .pill .dot{width:8px;height:8px;border-radius:50%}
  .pill.ok{color:var(--up)} .pill.ok .dot{background:var(--up)}
  .pill.warn{color:var(--cooling)} .pill.warn .dot{background:var(--cooling)}
  .pill.bad{color:var(--down)} .pill.bad .dot{background:var(--down)}
  .updated{color:var(--muted); font-size:12.5px; display:flex; align-items:center; gap:7px}
  .heartbeat{width:7px;height:7px;border-radius:50%;background:var(--accent);opacity:.9}
  .beat{animation:beat 2s ease-in-out infinite}
  @keyframes beat{0%,100%{opacity:.25;transform:scale(.85)}50%{opacity:1;transform:scale(1)}}

  .tabs{display:flex; gap:4px; border-bottom:1px solid var(--line); margin-bottom:22px}
  .tab-btn{font:inherit; font-size:13.5px; font-weight:600; color:var(--muted); background:none; border:none;
    border-bottom:2px solid transparent; padding:9px 14px; cursor:pointer; margin-bottom:-1px}
  .tab-btn:hover{color:var(--fg)}
  .tab-btn.on{color:var(--fg); border-bottom-color:var(--accent)}
  .tab-btn:focus-visible{outline:2px solid var(--accent); outline-offset:2px; border-radius:6px}
  .tab{display:none} .tab.on{display:block; animation:rise .18s ease both}

  .summary{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:24px}
  .stat{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px}
  .stat .k{color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.6px}
  .stat .v{font-size:22px; font-weight:680; margin-top:3px}
  .stat .v small{font-size:13px; color:var(--muted); font-weight:500}

  .provider{margin-bottom:24px}
  .phead{display:flex; align-items:center; gap:10px; flex-wrap:wrap; margin:0 2px 11px}
  .phead .pdot{width:10px;height:10px;border-radius:50%;flex:none}
  .phead.g-ok .pdot{background:var(--up); box-shadow:0 0 0 3px var(--up-bg)}
  .phead.g-warn .pdot{background:var(--cooling); box-shadow:0 0 0 3px var(--cooling-bg)}
  .phead.g-bad .pdot{background:var(--down); box-shadow:0 0 0 3px var(--down-bg)}
  .phead .pname{font-family:var(--mono); font-weight:650; font-size:15px}
  .ptype{font-size:10.5px; color:var(--muted); border:1px solid var(--line-2); border-radius:6px; padding:1px 7px; text-transform:uppercase; letter-spacing:.5px}
  .phead .pmeta{color:var(--muted); font-size:12px}
  .phead .pacts{margin-left:auto; display:flex; gap:6px}

  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:var(--radius); padding:16px 16px 14px;
    box-shadow:var(--shadow); position:relative; transition:transform .16s ease, border-color .16s ease; animation:rise .22s ease both}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .card.active{border-color:color-mix(in srgb,var(--accent) 55%, var(--line))}
  .card.active:before{content:"";position:absolute;inset:-1px;border-radius:var(--radius);box-shadow:0 0 0 1px var(--accent) inset, 0 0 22px var(--accent-bg);pointer-events:none;opacity:.55}
  .card .row1{display:flex; align-items:center; gap:9px; margin-bottom:8px}
  .sdot{width:11px;height:11px;border-radius:50%;flex:none}
  .s-up .sdot{background:var(--up); box-shadow:0 0 0 3px var(--up-bg)}
  .s-cooling .sdot{background:var(--cooling); box-shadow:0 0 0 3px var(--cooling-bg)}
  .s-down .sdot{background:var(--down); box-shadow:0 0 0 3px var(--down-bg)}
  .s-off .sdot{background:var(--off); box-shadow:0 0 0 3px var(--off-bg)}
  .cid{font-family:var(--mono); font-weight:600; font-size:14px}
  .state{font-size:11.5px; font-weight:700; letter-spacing:.5px; text-transform:uppercase; padding:2px 8px; border-radius:6px}
  .s-up .state{color:var(--up); background:var(--up-bg)}
  .s-cooling .state{color:var(--cooling); background:var(--cooling-bg)}
  .s-down .state{color:var(--down); background:var(--down-bg)}
  .s-off .state{color:var(--off); background:var(--off-bg)}
  .badge{font-size:10.5px; font-weight:700; letter-spacing:.6px; text-transform:uppercase; padding:2px 7px; border-radius:6px; border:1px solid var(--line-2); color:var(--muted)}
  .badge.active{color:var(--accent); border-color:color-mix(in srgb,var(--accent) 45%, var(--line-2)); background:var(--accent-bg)}
  .badge.pin{color:#9b7bff; border-color:color-mix(in srgb,#9b7bff 45%, var(--line-2))}
  .meta{color:var(--muted); font-size:12px; display:flex; align-items:center; gap:8px; margin:2px 0 10px}
  .host{font-family:var(--mono)}
  .reason{font-size:12.5px; padding:8px 10px; border-radius:9px; margin-bottom:11px; border:1px solid var(--line)}
  .reason.soft{background:var(--cooling-bg)} .reason.hard{background:var(--down-bg)}
  .reason .why{font-weight:650; text-transform:capitalize} .reason .err{color:var(--muted); word-break:break-word} .reason .cd{color:var(--muted)}
  .metrics{display:grid; grid-template-columns:repeat(3,1fr); gap:8px 12px; margin-bottom:12px}
  .metric .mk{color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.5px}
  .metric .mv{font-size:14px; font-weight:620; margin-top:1px}
  .actions{display:flex; gap:7px; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:11px}

  button.act{font:inherit; font-size:12.5px; font-weight:600; cursor:pointer; color:var(--fg); background:var(--panel-2);
    border:1px solid var(--line-2); border-radius:8px; padding:5px 11px; transition:background .14s ease, border-color .14s ease, transform .05s ease}
  button.act:hover{border-color:var(--accent); background:var(--accent-bg)}
  button.act:active{transform:translateY(1px)}
  button.act:disabled{opacity:.4; cursor:not-allowed}
  button.act.danger:hover{border-color:var(--down); background:var(--down-bg); color:var(--down)}
  button.act.primary{color:#fff; background:var(--accent); border-color:var(--accent)}
  button.act.primary:hover{filter:brightness(1.08)}
  @media (prefers-color-scheme: light){ button.act.primary{color:#fff} }

  /* editor cards */
  .ecard{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; box-shadow:var(--shadow); margin-bottom:12px}
  .ecard .er1{display:flex; align-items:center; gap:9px; margin-bottom:6px}
  .ecard .ename{font-family:var(--mono); font-weight:650; font-size:14.5px}
  .ecard .eacts{margin-left:auto; display:flex; gap:6px}
  .ecard .edesc{color:var(--muted); font-size:12.5px; line-height:1.5}
  .ecard .accs{margin-top:10px; display:flex; flex-direction:column; gap:7px}
  .accrow{display:flex; align-items:center; gap:9px; background:var(--panel-2); border:1px solid var(--line); border-radius:9px; padding:7px 10px; font-size:12.5px}
  .accrow .aname{font-family:var(--mono); font-weight:600}
  .accrow .aenv{color:var(--muted); font-family:var(--mono)}
  .accrow .aacts{margin-left:auto; display:flex; gap:6px}
  .keychip{font-size:10px; font-weight:700; padding:1px 6px; border-radius:5px; text-transform:uppercase}
  .keychip.ok{color:var(--up); background:var(--up-bg)} .keychip.no{color:var(--down); background:var(--down-bg)}
  .sect-head{display:flex; align-items:center; margin:0 2px 12px}
  .sect-head h2{font-size:15px; margin:0} .sect-head .spacer{flex:1}

  /* modals */
  .ovl{position:fixed; inset:0; background:rgba(4,7,11,.5); display:none; align-items:center; justify-content:center; z-index:30; backdrop-filter:blur(2px); padding:20px}
  .ovl.on{display:flex; animation:fade .12s ease both}
  @keyframes fade{from{opacity:0}to{opacity:1}}
  .modal{background:var(--panel); border:1px solid var(--line-2); border-radius:16px; box-shadow:var(--shadow); padding:24px 24px 20px; max-width:440px; width:100%; animation:rise .16s ease both}
  .modal.wide{max-width:720px; max-height:calc(100vh - 48px); overflow:auto; padding:26px 28px 22px}
  .modal h3{margin:0 0 10px; font-size:18px; font-weight:680; letter-spacing:.2px; display:flex; align-items:center; gap:10px}
  .modal h3 .idot{width:10px;height:10px;border-radius:50%; background:var(--cooling); flex:none}
  .modal h3.danger .idot{background:var(--down)}
  .modal p{margin:0 0 18px; color:var(--muted); font-size:14px; line-height:1.6; text-wrap:pretty}
  .modal p b{color:var(--fg); font-family:var(--mono); font-weight:600}
  .modal .row{display:flex; gap:10px; justify-content:flex-end; margin-top:16px}
  .modal .row button.act{font-size:14px; padding:9px 20px; border-radius:9px}
  .modal .ferr{color:var(--down); font-size:13px; margin:0 0 14px; display:none}
  .modal .ferr.on{display:block}

  .fld{margin-bottom:18px}
  .fld > label{display:block; font-size:13.5px; font-weight:600; margin-bottom:5px}
  .fld .hint{color:var(--muted); font-size:12.5px; line-height:1.5; margin-bottom:8px; text-wrap:pretty}
  .fld input[type=text], .fld select, .fld textarea{width:100%; font:inherit; font-size:14px; background:var(--panel-2);
    border:1px solid var(--line-2); border-radius:9px; padding:10px 12px; color:var(--fg);
    transition:border-color .14s ease, box-shadow .14s ease}
  .fld input[type=text]:focus, .fld select:focus, .fld textarea:focus{outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-bg)}
  .fld input.mono, .fld textarea{font-family:var(--mono)}
  .fld textarea{min-height:64px; resize:vertical}
  .fld select{appearance:none; -webkit-appearance:none; padding-right:36px; cursor:pointer;
    background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238b98a9' stroke-width='2.5' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='6 9 12 15 18 9'/%3E%3C/svg%3E");
    background-repeat:no-repeat; background-position:right 13px center}

  /* toggle: a full clickable row-card with a proper-sized switch on the right */
  .fld.toggle{display:flex; align-items:center; gap:16px; margin-bottom:12px; padding:13px 15px;
    background:var(--panel-2); border:1px solid var(--line); border-radius:11px; cursor:pointer;
    transition:border-color .14s ease, background .14s ease}
  .fld.toggle:hover{border-color:var(--line-2)}
  .fld.toggle .tgtxt{flex:1; min-width:0}
  .fld.toggle .tgtxt label{display:block; font-size:13.5px; font-weight:600; margin:0 0 3px; cursor:pointer}
  .fld.toggle .tgtxt .hint{margin:0}
  .sw{width:48px; height:28px; border-radius:999px; background:var(--line-2); position:relative; flex:none;
    transition:background .18s ease}
  .sw:after{content:""; position:absolute; top:3px; left:3px; width:22px; height:22px; border-radius:50%;
    background:#fff; box-shadow:0 1px 3px rgba(0,0,0,.35); transition:left .18s cubic-bezier(.2,0,0,1)}
  .sw.on{background:var(--accent)} .sw.on:after{left:23px}
  .fld.toggle:focus-visible{outline:none; border-color:var(--accent); box-shadow:0 0 0 3px var(--accent-bg)}
  .rows{display:flex; flex-direction:column; gap:6px}
  .kv{display:flex; gap:6px} .kv input{flex:1}
  .tags{display:flex; flex-wrap:wrap; gap:6px; align-items:center}
  .tag{display:inline-flex; align-items:center; gap:6px; background:var(--panel-2); border:1px solid var(--line-2); border-radius:7px; padding:3px 4px 3px 9px; font-size:12.5px; font-family:var(--mono)}
  .tag button{border:none;background:none;color:var(--muted);cursor:pointer;font-size:14px;line-height:1;padding:0 2px}
  .tag button:hover{color:var(--down)}
  .addrow{display:flex; gap:6px} .addrow input, .addrow select{flex:1}

  .empty{color:var(--muted); text-align:center; padding:50px 0}
  #tokenbar{display:none; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--down); border-radius:12px; padding:12px 14px; margin-bottom:16px}
  #tokenbar input{flex:1; font:inherit; background:var(--panel-2); border:1px solid var(--line-2); border-radius:8px; padding:7px 10px; color:var(--fg)}
  #toasts{position:fixed; right:18px; bottom:18px; display:flex; flex-direction:column; gap:8px; z-index:40}
  .toast{background:var(--panel); border:1px solid var(--line-2); border-left:3px solid var(--accent); border-radius:9px; padding:10px 14px; box-shadow:var(--shadow); font-size:13px; animation:rise .18s ease both}
  .toast.err{border-left-color:var(--down)} .toast.ok{border-left-color:var(--up)}
  @media (prefers-reduced-motion: reduce){*{animation:none !important; transition:none !important}}
</style>
</head>
<body>
  <header class="top">
    <div class="brand"><span class="logo"></span> llm-fusion</div>
    <span class="spacer"></span>
    <span id="overall" class="pill"><span class="dot"></span><span class="lbl">…</span></span>
    <span class="updated"><span id="hb" class="heartbeat"></span><span id="updated">connecting…</span></span>
  </header>

  <div class="tabs" role="tablist" aria-label="Panel sections">
    <button class="tab-btn on" data-tab="monitor" id="tabbtn-monitor" role="tab" aria-selected="true" aria-controls="tab-monitor">Monitor</button>
    <button class="tab-btn" data-tab="providers" id="tabbtn-providers" role="tab" aria-selected="false" aria-controls="tab-providers" tabindex="-1">Providers</button>
    <button class="tab-btn" data-tab="models" id="tabbtn-models" role="tab" aria-selected="false" aria-controls="tab-models" tabindex="-1">Models</button>
    <button class="tab-btn" data-tab="settings" id="tabbtn-settings" role="tab" aria-selected="false" aria-controls="tab-settings" tabindex="-1">Settings</button>
  </div>

  <form id="tokenbar" onsubmit="return false;">
    <span>Auth required — paste the admin/panel token:</span>
    <input id="tokenin" type="password" placeholder="Bearer token" autocomplete="off" />
    <button class="act" id="tokensave" type="submit">Save</button>
  </form>

  <div id="tab-monitor" class="tab on" role="tabpanel" aria-labelledby="tabbtn-monitor" tabindex="0">
    <section id="summary" class="summary"></section>
    <section id="providers"></section>
    <div id="empty-mon" class="empty" style="display:none">No providers reported.</div>
  </div>

  <div id="tab-providers" class="tab" role="tabpanel" aria-labelledby="tabbtn-providers" tabindex="0">
    <div class="sect-head"><h2>Providers &amp; accounts</h2><span class="spacer"></span><button class="act primary" id="add-provider">+ Add provider</button></div>
    <div id="providers-editor"></div>
    <div id="empty-prov" class="empty" style="display:none">No providers configured.</div>
  </div>

  <div id="tab-models" class="tab" role="tabpanel" aria-labelledby="tabbtn-models" tabindex="0">
    <div class="sect-head"><h2>Models</h2><span class="spacer"></span><button class="act primary" id="add-model">+ Create model</button></div>
    <div id="models-editor"></div>
    <div id="empty-models" class="empty" style="display:none">No models configured.</div>
  </div>

  <div id="tab-settings" class="tab" role="tabpanel" aria-labelledby="tabbtn-settings" tabindex="0">
    <div class="sect-head"><h2>Global settings</h2><span class="spacer"></span><button class="act danger" id="restart-btn">Restart service</button></div>
    <div id="settings-editor"></div>
  </div>

  <div id="ovl" class="ovl">
    <div class="modal" role="dialog" aria-modal="true" aria-labelledby="ovl-title-txt">
      <h3 id="ovl-title"><span class="idot"></span><span id="ovl-title-txt">Confirm</span></h3>
      <p id="ovl-msg"></p>
      <div class="row"><button class="act" id="ovl-cancel" type="button">Cancel</button><button class="act danger" id="ovl-ok" type="button">Confirm</button></div>
    </div>
  </div>

  <div id="fovl" class="ovl">
    <div class="modal wide" role="dialog" aria-modal="true" aria-labelledby="fovl-title">
      <h3><span id="fovl-title">Form</span></h3>
      <p id="fovl-err" class="ferr"></p>
      <div id="fovl-body"></div>
      <div class="row"><button class="act" id="fovl-cancel" type="button">Cancel</button><button class="act primary" id="fovl-save" type="button">Save</button></div>
    </div>
  </div>

  <div id="toasts" role="status" aria-live="polite" aria-atomic="false"></div>

<script>
(function(){
  "use strict";
  var TOKEN_KEY = "fusion_panel_token", POLL_MS = 3000;
  var last = null, busy = {}, groups = {}, sum = null, since = Date.now();
  var cfg = null, activeTab = "monitor";

  function tok(){ try { return localStorage.getItem(TOKEN_KEY) || ""; } catch(e){ return ""; } }
  function setTok(v){ try { localStorage.setItem(TOKEN_KEY, v); } catch(e){} }
  function authHeaders(extra){ var h = extra || {}; var t = tok(); if (t) h["authorization"] = "Bearer " + t; return h; }
  function el(tag, cls, text){ var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  function setText(n, v){ if (n.textContent !== v) n.textContent = v; }
  function setClass(n, v){ if (n.className !== v) n.className = v; }
  function show(n, on){ var d = on ? "" : "none"; if (n.style.display !== d) n.style.display = d; }
  function esc(s){ return String(s).replace(/[&<>"]/g, function(c){ return { "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]; }); }
  function fmtInt(n){ return (n == null ? 0 : n).toLocaleString("en-US"); }
  function pct(f, t){ if (!t) return "—"; return Math.round((1 - f / t) * 100) + "%"; }
  function rel(ts){ if (!ts) return "—"; var s = Math.max(0, Math.round((Date.now()-ts)/1000));
    if (s<60) return s+"s ago"; var m=Math.floor(s/60); if (m<60) return m+"m ago"; var h=Math.floor(m/60); if (h<24) return h+"h ago"; return Math.floor(h/24)+"d ago"; }
  function secs(ms){ if (ms==null) return ""; var s=Math.ceil(ms/1000); return s>0 ? s+"s" : "0s"; }
  function toast(msg, kind){ var box=document.getElementById("toasts"); var t=el("div","toast"+(kind?" "+kind:""),msg); box.appendChild(t);
    setTimeout(function(){ t.style.opacity="0"; t.style.transform="translateY(6px)"; },2600); setTimeout(function(){ if(t.parentNode) t.parentNode.removeChild(t); },3000); }
  function showTokenBar(s){ document.getElementById("tokenbar").style.display = s ? "flex" : "none"; }

  // --- fetch helpers ------------------------------------------------------
  function jget(path){
    return fetch(path, { headers: authHeaders(), cache:"no-store" }).then(function(r){
      if (r.status===401){ showTokenBar(true); throw new Error("auth"); }
      showTokenBar(false); if (!r.ok) throw new Error("HTTP "+r.status); return r.json();
    });
  }
  function jsend(method, path, body){
    var opt = { method: method, headers: authHeaders(body ? { "content-type":"application/json" } : {}) };
    if (body) opt.body = JSON.stringify(body);
    return fetch(path, opt).then(function(r){
      if (r.status===401){ showTokenBar(true); throw new Error("auth required"); }
      return r.json().then(function(j){ if (!r.ok) throw new Error(j && j.error ? j.error : "HTTP "+r.status); return j; },
        function(){ if (!r.ok) throw new Error("HTTP "+r.status); return {}; });
    });
  }

  // --- confirm modal ------------------------------------------------------
  var pendingYes = null;
  var confirmPrevFocus = null;
  function confirmAction(title, msgHtml, danger, onYes){
    confirmPrevFocus = document.activeElement;
    document.getElementById("ovl-title-txt").textContent = title;
    document.getElementById("ovl-title").className = danger ? "danger" : "";
    document.getElementById("ovl-msg").innerHTML = msgHtml;
    var ok = document.getElementById("ovl-ok"); ok.textContent = title; ok.className = danger ? "act danger" : "act";
    pendingYes = onYes; document.getElementById("ovl").classList.add("on");
    document.getElementById("ovl-cancel").focus();
  }
  function closeConfirm(){ document.getElementById("ovl").classList.remove("on"); pendingYes = null;
    if(confirmPrevFocus&&confirmPrevFocus.focus) confirmPrevFocus.focus(); confirmPrevFocus=null; }
  document.getElementById("ovl-cancel").onclick = closeConfirm;
  document.getElementById("ovl-ok").onclick = function(){ var f=pendingYes; closeConfirm(); if (f) f(); };
  document.getElementById("ovl").onclick = function(e){ if (e.target===this) closeConfirm(); };
  document.getElementById("ovl").addEventListener("keydown", function(e){ trapTab(this, e); });

  // --- monitor tab (health) ----------------------------------------------
  function allAccounts(provs){ var out=[]; provs.forEach(function(p){ p.accounts.forEach(function(a){ out.push(a); }); }); return out; }
  function counts(items){ var c={up:0,cooling:0,down:0,off:0,reqs:0,fails:0};
    items.forEach(function(a){ if(a.state==="up")c.up++; else if(a.state==="cooling")c.cooling++; else if(a.state==="down")c.down++; else if(a.state==="off")c.off++; c.reqs+=a.totalRequests; c.fails+=a.totalFailures; }); return c; }
  function overall(provs){ var o=document.getElementById("overall"); var c=counts(allAccounts(provs));
    setClass(o,"pill "+(c.down?"bad":c.cooling?"warn":"ok")); setText(o.querySelector(".lbl"), c.down?(c.down+" down"):c.cooling?(c.cooling+" cooling"):(c.up+" healthy")); }
  function ensureSummary(){ if (sum) return; var box=document.getElementById("summary"); sum={};
    [["providers","providers"],["accounts","accounts"],["healthy","healthy"],["cooling","cooling"],["down / off","downoff"]].forEach(function(d){
      var s=el("div","stat"); s.appendChild(el("div","k",d[0])); var v=el("div","v num"); s.appendChild(v); box.appendChild(s); sum[d[1]]=v; });
    var s=el("div","stat"); s.appendChild(el("div","k","requests")); var v=el("div","v num"); var main=document.createTextNode(""); var sm=el("small"); v.appendChild(main); v.appendChild(sm); s.appendChild(v); box.appendChild(s); sum.reqMain=main; sum.reqSub=sm; }
  function updateSummary(provs){ ensureSummary(); var items=allAccounts(provs); var c=counts(items);
    setText(sum.providers,String(provs.length)); setText(sum.accounts,String(items.length)); setText(sum.healthy,String(c.up)); setText(sum.cooling,String(c.cooling)); setText(sum.downoff,String(c.down+c.off));
    var r=fmtInt(c.reqs); if(sum.reqMain.nodeValue!==r) sum.reqMain.nodeValue=r; setText(sum.reqSub," "+pct(c.fails,c.reqs)+" ok"); }
  function mkBtn(label, cls, disabled, fn){ var b=el("button",cls,label); b.disabled=disabled; b.onclick=fn; return b; }

  function buildCard(){
    var root=el("div","card"); var r1=el("div","row1"); r1.appendChild(el("span","sdot"));
    var cid=el("span","cid"); r1.appendChild(cid); var state=el("span","state"); r1.appendChild(state);
    var sp=el("span"); sp.style.flex="1"; r1.appendChild(sp); var activeBadge=el("span","badge active","active"); r1.appendChild(activeBadge);
    var pinBadge=el("span","badge pin","pinned"); r1.appendChild(pinBadge); root.appendChild(r1);
    var meta=el("div","meta"); var host=el("span","host mono"); var nokey=el("span","badge","no key"); meta.appendChild(host); meta.appendChild(nokey); root.appendChild(meta);
    var reason=el("div","reason"); var why=el("span","why"); var sep=document.createTextNode(""); var errEl=el("span","err"); var cd=el("div","cd");
    reason.appendChild(why); reason.appendChild(sep); reason.appendChild(errEl); reason.appendChild(cd); root.appendChild(reason);
    var metrics=el("div","metrics"); function metric(k){ var m=el("div","metric"); m.appendChild(el("div","mk",k)); var v=el("div","mv num"); m.appendChild(v); metrics.appendChild(m); return v; }
    var mReq=metric("requests"),mFail=metric("failures"),mSucc=metric("success"),mOk=metric("last ok"),mBad=metric("last fail"),mLat=metric("latency"); root.appendChild(metrics);
    var actions=el("div","actions"); root.appendChild(actions); var actKey="";
    function update(a){
      setClass(root,"card s-"+a.state+(a.active?" active":"")); setText(cid,a.id); setText(state,a.state);
      show(activeBadge,!!a.active); show(pinBadge,!!a.pinned); setText(host,a.host); show(nokey,!a.hasKey);
      var isR=(a.state==="cooling"||a.state==="down"||a.state==="off"); show(reason,isR);
      if (isR){ var hard=a.state==="down"; setClass(reason,"reason "+(hard?"hard":a.state==="off"?"":"soft")); setText(why,a.reason||(a.state==="off"?"manual":a.state));
        var he=!!a.lastError; show(errEl,he); var s=he?" · ":""; if(sep.nodeValue!==s) sep.nodeValue=s; if(he) setText(errEl,a.lastError);
        if (a.parked) setText(cd,"parked — manual reset only"); else if (a.cooldownRemainingMs!=null&&a.cooldownRemainingMs>0) setText(cd,(hard?"recheck in ":"probe in ")+secs(a.cooldownRemainingMs)); else if(a.state!=="off") setText(cd,"probing on next request"); else setText(cd,""); }
      setText(mReq,fmtInt(a.totalRequests)); setText(mFail,fmtInt(a.totalFailures)); setText(mSucc,pct(a.totalFailures,a.totalRequests));
      setText(mOk,rel(a.lastSuccessAt)); setText(mBad,rel(a.lastFailureAt)); setText(mLat,a.lastLatencyMs!=null?a.lastLatencyMs+"ms":"—");
      var b=!!busy[a.id]; var key=a.state+"|"+(a.active?1:0)+"|"+(a.pinned?1:0)+"|"+(b?1:0);
      if (key!==actKey){ actKey=key; actions.textContent="";
        if (a.state==="off") actions.appendChild(mkBtn("Enable","act",b,function(){ actAccount(a.id,"enable"); })); else actions.appendChild(mkBtn("Disable","act danger",b,function(){ actAccount(a.id,"disable"); }));
        if (a.state==="down"||a.state==="cooling") actions.appendChild(mkBtn("Reset","act",b,function(){ actAccount(a.id,"reset"); }));
        if (!a.active&&a.state!=="off") actions.appendChild(mkBtn("Make active","act",b,function(){ actAccount(a.id,"pin"); }));
        if (a.pinned) actions.appendChild(mkBtn("Unpin","act",b,function(){ actAccount(a.id,"unpin"); })); }
    }
    return { root: root, update: update };
  }
  function buildGroup(){
    var section=el("div","provider"); var head=el("div","phead"); head.appendChild(el("span","pdot"));
    var pname=el("span","pname"); head.appendChild(pname); var ptype=el("span","ptype"); head.appendChild(ptype);
    var pmeta=el("span","pmeta"); head.appendChild(pmeta); section.appendChild(head);
    var grid=el("div","grid"); section.appendChild(grid);
    return { section:section, head:head, pname:pname, ptype:ptype, pmeta:pmeta, grid:grid, cards:{} };
  }
  function updateGroupHead(g,p){ var c=counts(p.accounts); setClass(g.head,"phead "+(c.down?"g-bad":c.cooling?"g-warn":"g-ok"));
    setText(g.pname,p.id); setText(g.ptype,p.type); var act=p.activeId?("active: "+p.activeId):"no active account";
    setText(g.pmeta,p.accounts.length+" account"+(p.accounts.length===1?"":"s")+" · "+act); }
  function renderMonitor(payload){
    if (!payload) return; var provs=payload.providers||[]; show(document.getElementById("empty-mon"),provs.length===0);
    overall(provs); updateSummary(provs); var container=document.getElementById("providers"); var seenG={};
    provs.forEach(function(p){ seenG[p.id]=true; var g=groups[p.id]; if(!g){ g=buildGroup(); groups[p.id]=g; container.appendChild(g.section); } updateGroupHead(g,p);
      var seenA={}; p.accounts.forEach(function(a){ seenA[a.id]=true; var card=g.cards[a.id]; if(!card){ card=buildCard(); g.cards[a.id]=card; g.grid.appendChild(card.root); } card.update(a); });
      Object.keys(g.cards).forEach(function(id){ if(!seenA[id]){ if(g.cards[id].root.parentNode) g.grid.removeChild(g.cards[id].root); delete g.cards[id]; } }); });
    Object.keys(groups).forEach(function(id){ if(!seenG[id]){ if(groups[id].section.parentNode) container.removeChild(groups[id].section); delete groups[id]; } });
  }
  function actAccount(id, action){
    if (busy[id]) return;
    var msgs={ disable:"Disable account <b>"+esc(id)+"</b>? It will stop serving requests until you enable it again.",
      enable:"Enable account <b>"+esc(id)+"</b> so it can serve requests again?",
      reset:"Reset account <b>"+esc(id)+"</b> back to healthy (clear its cooldown / failure state)?",
      pin:"Make account <b>"+esc(id)+"</b> the active one for its provider?", unpin:"Clear the pinned active account for this provider?" };
    var cap=action.charAt(0).toUpperCase()+action.slice(1);
    confirmAction(cap, msgs[action]||("Confirm "+esc(action)+"?"), action==="disable", function(){
      busy[id]=true; renderMonitor(last);
      jsend("POST","admin/connectors/"+encodeURIComponent(id)+"/"+action).then(function(j){ last=j; toast(id+" · "+action,"ok"); renderMonitor(last); })
        .catch(function(e){ toast(String(e.message||e),"err"); }).finally(function(){ delete busy[id]; renderMonitor(last); });
    });
  }

  // --- config editor (Providers + Models tabs) ---------------------------
  function loadConfig(){ return jget("admin/config").then(function(j){ cfg=j; renderProviders(); renderModels(); renderSettings(); }); }
  function groupIds(){ return cfg && cfg.providers ? Object.keys(cfg.providers) : []; }
  function modelNames(){ return cfg && cfg.models ? Object.keys(cfg.models) : []; }
  function reloadConfigSoon(){ setTimeout(loadConfig, 400); } // let the file watcher hot-reload first

  // Live upstream model catalog per provider group (for the no-typo picker).
  // Only a NON-EMPTY catalog is cached: an empty result (genuine-empty OR a soft
  // failure — the server returns 200 + {models:[],note} so the form never blocks)
  // is treated as a cache miss, so a transient outage/unsaved-key retries on the
  // next form open instead of hiding suggestions for the whole panel session.
  var provModelsCache={};
  function fetchProviderModels(groupId, cb){
    if(!groupId){ cb([]); return; }
    var cached=provModelsCache[groupId];
    if(cached&&cached.length){ cb(cached); return; }
    jget("admin/config/providers/"+encodeURIComponent(groupId)+"/models").then(function(j){ var m=(j&&j.models)||[]; if(m.length) provModelsCache[groupId]=m; cb(m); }).catch(function(){ cb([]); });
  }

  function saveModel(name, obj, done){ jsend("PUT","admin/config/models/"+encodeURIComponent(name), obj).then(function(){ toast("model '"+name+"' saved","ok"); reloadConfigSoon(); if(done)done(true); })
    .catch(function(e){ if(done)done(false,String(e.message||e)); }); }
  function deleteModel(name){ jsend("DELETE","admin/config/models/"+encodeURIComponent(name)).then(function(){ toast("model '"+name+"' deleted","ok"); reloadConfigSoon(); }).catch(function(e){ toast(String(e.message||e),"err"); }); }
  function saveProvider(id, obj, done){ jsend("PUT","admin/config/providers/"+encodeURIComponent(id), obj).then(function(){ toast("provider '"+id+"' saved","ok"); reloadConfigSoon(); if(done)done(true); })
    .catch(function(e){ if(done)done(false,String(e.message||e)); }); }
  function deleteProvider(id){ jsend("DELETE","admin/config/providers/"+encodeURIComponent(id)).then(function(){ toast("provider '"+id+"' deleted","ok"); reloadConfigSoon(); }).catch(function(e){ toast(String(e.message||e),"err"); }); }

  // ---- form modal + field builders ----
  var formSave = null;
  var formPrevFocus = null;
  var dlSeq = 0; // unique field/datalist ids
  var FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
  // Keep Tab inside an open modal so keyboard focus never escapes behind the overlay.
  function trapTab(container, e){ if(e.key!=="Tab") return; var f=container.querySelectorAll(FOCUSABLE); if(!f.length) return; var first=f[0], last=f[f.length-1];
    if(e.shiftKey && document.activeElement===first){ e.preventDefault(); last.focus(); }
    else if(!e.shiftKey && document.activeElement===last){ e.preventDefault(); first.focus(); } }
  function openForm(title, buildBody, onSave){
    formPrevFocus = document.activeElement;
    document.getElementById("fovl-title").textContent = title;
    var err=document.getElementById("fovl-err"); err.className="ferr"; err.textContent="";
    var body=document.getElementById("fovl-body"); body.textContent=""; buildBody(body);
    formSave = onSave; document.getElementById("fovl").classList.add("on");
    var first=body.querySelector(FOCUSABLE); if(first) first.focus();
  }
  function closeForm(){ document.getElementById("fovl").classList.remove("on"); formSave=null;
    if(formPrevFocus&&formPrevFocus.focus) formPrevFocus.focus(); formPrevFocus=null; }
  function formError(msg){ var e=document.getElementById("fovl-err"); e.textContent=msg; e.className="ferr on"; }
  document.getElementById("fovl-cancel").onclick = closeForm;
  document.getElementById("fovl").onclick = function(e){ if (e.target===this) closeForm(); };
  document.getElementById("fovl").addEventListener("keydown", function(e){ trapTab(this, e); });
  document.getElementById("fovl-save").onclick = function(){ if (formSave) formSave(); };

  function fld(label, hint){ var f=el("div","fld"); var l=el("label",null,label); f.appendChild(l); f._label=l; if(hint){ var h=el("div","hint",hint); h.id="fh"+(++dlSeq); f._hintId=h.id; f.appendChild(h); } return f; }
  // Always create the datalist (even when empty) and return a fill(list) updater,
  // so the provider catalog can be refreshed IN PLACE without rebuilding the field
  // (which would destroy focus / cursor / a half-typed tag).
  function attachSuggest(input, f, suggestions){ var did="dl"+(++dlSeq); var dl=el("datalist"); dl.id=did; input.setAttribute("list",did); f.appendChild(dl);
    function fill(list){ dl.textContent=""; (list||[]).forEach(function(s){ var o=el("option"); o.value=s; dl.appendChild(o); }); }
    fill(suggestions); return fill; }
  function fText(label, hint, value, mono, suggestions){ var f=fld(label,hint); var i=el("input"); i.type="text"; i.id="fi"+(++dlSeq); if(f._label)f._label.htmlFor=i.id; if(f._hintId)i.setAttribute("aria-describedby",f._hintId); if(mono) i.className="mono"; i.value=value==null?"":value; f._setSuggest=attachSuggest(i,f,suggestions); f.appendChild(i); f._get=function(){ return i.value.trim(); }; return f; }
  // Numeric field: text input (so it can be blank = "unset") that parses to a
  // number on read. Returns undefined when blank, or NaN when non-numeric so the
  // caller can reject it. inputmode=numeric brings up the number keypad on mobile.
  function fNum(label, hint, value){ var f=fld(label,hint); var i=el("input"); i.type="text"; i.setAttribute("inputmode","numeric"); i.className="mono num"; i.id="fi"+(++dlSeq); if(f._label)f._label.htmlFor=i.id; if(f._hintId)i.setAttribute("aria-describedby",f._hintId); i.value=(value==null?"":String(value)); f.appendChild(i);
    f._get=function(){ var v=i.value.trim(); if(v==="") return undefined; var n=Number(v); return isFinite(n)?n:NaN; }; return f; }
  function fSelect(label, hint, value, options){ var f=fld(label,hint); var s=el("select"); s.id="fi"+(++dlSeq); if(f._label)f._label.htmlFor=s.id; if(f._hintId)s.setAttribute("aria-describedby",f._hintId); options.forEach(function(o){ var op=el("option",null,o.label||o); op.value=(o.value!=null?o.value:o); if((o.value!=null?o.value:o)===value) op.selected=true; s.appendChild(op); }); f.appendChild(s); f._get=function(){ return s.value; }; return f; }
  function fToggle(label, hint, value){ var f=el("div","fld toggle"); var d=el("div","tgtxt");
    var lab=el("label",null,label); lab.id="tgl"+(++dlSeq); d.appendChild(lab);
    var hid=""; if(hint){ hid="tgh"+dlSeq; var hn=el("div","hint",hint); hn.id=hid; d.appendChild(hn); } f.appendChild(d);
    var sw=el("div","sw"+(value?" on":"")); sw.setAttribute("aria-hidden","true"); f.appendChild(sw);
    f.setAttribute("role","switch"); f.setAttribute("tabindex","0"); f.setAttribute("aria-checked", value?"true":"false"); f.setAttribute("aria-labelledby", lab.id); if(hid) f.setAttribute("aria-describedby", hid);
    function set(on){ sw.classList.toggle("on", !!on); f.setAttribute("aria-checked", on?"true":"false"); }
    f.onclick=function(){ set(!sw.classList.contains("on")); };
    f.addEventListener("keydown", function(e){ if(e.key===" "||e.key==="Spacebar"||e.key==="Enter"){ e.preventDefault(); set(!sw.classList.contains("on")); } });
    f._get=function(){ return sw.classList.contains("on"); }; return f; }
  function fTags(label, hint, arr, suggestions){ var f=fld(label,hint); var box=el("div","tags"); var vals=(arr||[]).slice(); var sugg=(suggestions||[]).slice(); var curFill=null;
    function draw(){ box.textContent=""; vals.forEach(function(v,i){ var t=el("span","tag"); t.appendChild(document.createTextNode(v)); var x=el("button",null,"×"); x.type="button"; x.onclick=function(){ vals.splice(i,1); draw(); }; t.appendChild(x); box.appendChild(t); });
      var add=el("span","addrow"); var inp=el("input"); inp.type="text"; inp.className="mono"; inp.placeholder="add…"; inp.setAttribute("aria-label","Add to "+label); curFill=attachSuggest(inp,add,sugg);
      inp.onkeydown=function(e){ if(e.key==="Enter"){ e.preventDefault(); var v=inp.value.trim(); if(v){ vals.push(v); draw(); } } };
      var b=el("button","act",""); b.type="button"; b.textContent="Add"; b.onclick=function(){ var v=inp.value.trim(); if(v){ vals.push(v); draw(); } };
      add.appendChild(inp); add.appendChild(b); box.appendChild(add); }
    draw(); f.appendChild(box); f._get=function(){ return vals.slice(); };
    f._setSuggest=function(list){ sugg=(list||[]).slice(); if(curFill) curFill(sugg); }; return f; }
  function fKV(label, hint, obj){ var f=fld(label,hint); var wrap=el("div","rows"); var pairs=Object.keys(obj||{}).map(function(k){ return [k,obj[k]]; });
    function draw(){ wrap.textContent=""; pairs.forEach(function(p,i){ var row=el("div","kv"); var k=el("input"); k.type="text"; k.className="mono"; k.placeholder="from"; k.setAttribute("aria-label",label+" key"); k.value=p[0]; var v=el("input"); v.type="text"; v.className="mono"; v.placeholder="to"; v.setAttribute("aria-label",label+" value"); v.value=p[1];
      k.oninput=function(){ p[0]=k.value; }; v.oninput=function(){ p[1]=v.value; }; var x=el("button","act",""); x.type="button"; x.textContent="×"; x.onclick=function(){ pairs.splice(i,1); draw(); };
      row.appendChild(k); row.appendChild(v); row.appendChild(x); wrap.appendChild(row); });
      var add=el("button","act",""); add.type="button"; add.textContent="+ Add mapping"; add.onclick=function(){ pairs.push(["",""]); draw(); }; wrap.appendChild(add); }
    draw(); f.appendChild(wrap); f._get=function(){ var o={}; pairs.forEach(function(p){ if(p[0].trim()) o[p[0].trim()]=p[1].trim(); }); return o; }; return f; }

  // Reveal/hide a container of sub-fields as a toggle flips. Layers on top of
  // fToggle's own handlers (its onclick fires first, then this), so reading the
  // toggle state here already sees the new value. Also drives initial visibility.
  function bindReveal(tg, container){
    function sync(){ container.style.display = tg._get() ? "" : "none"; }
    tg.addEventListener("click", sync);
    tg.addEventListener("keydown", function(e){ if(e.key===" "||e.key==="Spacebar"||e.key==="Enter") setTimeout(sync,0); });
    sync();
  }
  // A slightly-inset container for a toggle's dependent sub-fields.
  function subGroup(){ var d=el("div"); d.style.margin="0 0 4px 2px"; d.style.paddingLeft="12px"; d.style.borderLeft="2px solid var(--line)"; return d; }

  // ---- Providers tab ----
  function renderProviders(){
    var box=document.getElementById("providers-editor"); box.textContent="";
    var provs=cfg && cfg.providers ? cfg.providers : {};
    var ids=Object.keys(provs); show(document.getElementById("empty-prov"), ids.length===0);
    ids.forEach(function(id){ var p=provs[id]; var c=el("div","ecard");
      var r=el("div","er1"); r.appendChild(el("span","ename",id)); r.appendChild(el("span","ptype",p.type));
      var acts=el("div","eacts"); acts.appendChild(mkBtn("+ Account","act",false,function(){ accountForm(id,null); }));
      acts.appendChild(mkBtn("Edit","act",false,function(){ providerForm(id,p); }));
      acts.appendChild(mkBtn("Delete","act danger",false,function(){ confirmAction("Delete","Delete provider <b>"+esc(id)+"</b> and all its accounts? Models bound to it will fail to load until reassigned.",true,function(){ deleteProvider(id); }); }));
      r.appendChild(acts); c.appendChild(r);
      c.appendChild(el("div","edesc",(p.base_url||"(no base_url)")+" · "+(p.accounts?p.accounts.length:0)+" account(s)"));
      var accs=el("div","accs"); (p.accounts||[]).forEach(function(a){ var ar=el("div","accrow");
        ar.appendChild(el("span","aname",a.id)); ar.appendChild(el("span","aenv",a.api_key_env));
        var known=cfg.envKnown && cfg.envKnown[a.api_key_env]; ar.appendChild(el("span","keychip "+(known?"ok":"no"),known?"key set":"no key"));
        var aa=el("div","aacts"); aa.appendChild(mkBtn("Edit","act",false,function(){ accountForm(id,a); }));
        aa.appendChild(mkBtn("Delete","act danger",false,function(){ confirmAction("Delete","Delete account <b>"+esc(a.id)+"</b> from provider <b>"+esc(id)+"</b>?",true,function(){ deleteAccount(id,a.id); }); }));
        ar.appendChild(aa); accs.appendChild(ar); });
      c.appendChild(accs); box.appendChild(c); });
  }
  // Adding a NEW provider group when one already exists makes previously-unbound
  // models ambiguous — bind them to the current sole group first, then continue.
  function bindUnboundModelsThen(groupId, next){
    var models=cfg&&cfg.models?cfg.models:{}; var unbound=Object.keys(models).filter(function(n){ return !models[n].provider; });
    if(!unbound.length){ next(); return; }
    var i=0; (function step(){ if(i>=unbound.length){ toast("bound "+unbound.length+" model(s) to "+groupId,"ok"); next(); return; }
      var n=unbound[i++]; var m=Object.assign({},models[n],{provider:groupId});
      jsend("PUT","admin/config/models/"+encodeURIComponent(n),m).then(step,function(e){ next(String(e.message||e)); }); })();
  }
  function providerForm(id, existing){
    var fId, fType, fBase, fAccId, fAccEnv;
    openForm(existing?("Edit provider "+id):"Add provider", function(body){
      fId=fText("Provider id","A short name you reference in models (e.g. ollama-cloud, openrouter).", existing?id:"", true);
      if (existing){ fId.querySelector("input").disabled=true; }
      fType=fSelect("Type","ollama = Ollama Cloud (adds capability discovery + native vision). openai-compat = any OpenAI-compatible API (OpenRouter, DeepInfra, Together, …).", existing?existing.type:"ollama",[{label:"ollama",value:"ollama"},{label:"openai-compat",value:"openai-compat"}]);
      fBase=fText("Base URL","The provider's API root, e.g. https://ollama.com or https://openrouter.ai/api/v1.", existing?existing.base_url:"", true);
      body.appendChild(fId); body.appendChild(fType); body.appendChild(fBase);
      if (!existing){
        var note=el("div","hint"); note.style.margin="6px 2px 10px"; note.textContent="Its first account (add more later):"; body.appendChild(note);
        fAccId=fText("Account id","Unique across ALL providers (e.g. ollama-1).","",true);
        fAccEnv=fText("API key env var","Env var holding the key — never the key itself (e.g. OLLAMA_API_KEY).","",true);
        body.appendChild(fAccId); body.appendChild(fAccEnv);
      }
    }, function(){
      var nid=fId._get(); if(!nid){ formError("Provider id is required."); return; }
      var base=fBase._get();
      if (existing){
        var obj=Object.assign({},existing,{type:fType._get(),base_url:base||undefined});
        saveProvider(nid, obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
        return;
      }
      var accId=fAccId._get(), accEnv=fAccEnv._get();
      if(!accId||!accEnv){ formError("The first account's id and API key env var are required."); return; }
      var nobj={ type:fType._get(), accounts:[{ id:accId, api_key_env:accEnv }] }; if(base) nobj.base_url=base;
      var ids=groupIds();
      var doSave=function(){ saveProvider(nid, nobj, function(ok,err){ if(ok) closeForm(); else formError(err); }); };
      if (ids.length===1) bindUnboundModelsThen(ids[0], function(err){ if(err){ formError(err); return; } doSave(); });
      else doSave();
    });
  }
  function accountForm(provId, existing){
    var p=cfg.providers[provId]; var fId,fEnv,fBase,fMap,f403,fQuota;
    openForm(existing?("Edit account "+existing.id):("Add account to "+provId), function(body){
      fId=fText("Account id","Unique across ALL providers (e.g. ollama-1). Used in the monitor + controls.", existing?existing.id:"", true);
      if(existing){ fId.querySelector("input").disabled=true; }
      fEnv=fText("API key env var","Name of the environment variable holding this account's key — never the key itself (e.g. OLLAMA_API_KEY, OLLAMA_API_KEY_2).", existing?existing.api_key_env:"", true);
      fBase=fText("Base URL override (optional)","Leave blank to use the provider's base URL. Set only if this account uses a different endpoint.", existing?existing.base_url:"", true);
      fMap=fKV("Model-id map (optional)","Map a logical model id (used in Models) to THIS provider's id — e.g. glm-5.2 → z-ai/glm-4.6. Needed when the provider names models differently than Ollama.", existing?existing.model_map:{});
      f403=fSelect("On HTTP 403","passthrough = treat as a client error (account stays healthy). down = mark the account down (some gateways use 403 for no-credits).", existing?existing.treat_403_as:"passthrough",[{label:"passthrough",value:"passthrough"},{label:"down",value:"down"}]);
      fQuota=fTags("Quota markers (optional)","Lowercase phrases in a 429 body that mean the account is OUT OF QUOTA (mark it down) rather than a brief rate-limit — e.g. insufficient, out of credit.", existing?existing.quota_markers:[]);
      body.appendChild(fId); body.appendChild(fEnv); body.appendChild(fBase); body.appendChild(fMap); body.appendChild(f403); body.appendChild(fQuota);
    }, function(){
      var nid=fId._get(), env=fEnv._get(); if(!nid||!env){ formError("Account id and API key env var are required."); return; }
      var acc={ id:nid, api_key_env:env }; var b=fBase._get(); if(b) acc.base_url=b;
      var m=fMap._get(); if(Object.keys(m).length) acc.model_map=m; var q=fQuota._get(); if(q.length) acc.quota_markers=q;
      var t=f403._get(); if(t!=="passthrough") acc.treat_403_as=t;
      var accounts=(p.accounts||[]).slice(); var idx=accounts.map(function(a){return a.id;}).indexOf(nid);
      if(idx>=0) accounts[idx]=acc; else accounts.push(acc);
      var obj=Object.assign({},p,{accounts:accounts});
      saveProvider(provId, obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
  function deleteAccount(provId, accId){ var p=cfg.providers[provId]; var accounts=(p.accounts||[]).filter(function(a){ return a.id!==accId; });
    if(accounts.length===0){ toast("a provider must keep at least one account — delete the provider instead","err"); return; }
    saveProvider(provId, Object.assign({},p,{accounts:accounts})); }
  document.getElementById("add-provider").onclick=function(){ providerForm(null,null); };

  // ---- Models tab ----
  function strategySummary(m){ if(m.strategy==="single") return "single → "+m.target;
    if(m.strategy==="failover") return "failover → "+(m.chain||[]).join(", ");
    if(m.strategy==="fusion") return "fusion · panel ["+(m.panel||[]).join(", ")+"] · judge "+m.judge+" · synth "+m.synth;
    if(m.strategy==="smart") return "smart · router "+m.router; return m.strategy; }
  function renderModels(){
    var box=document.getElementById("models-editor"); box.textContent="";
    var models=cfg && cfg.models ? cfg.models : {}; var names=Object.keys(models); show(document.getElementById("empty-models"),names.length===0);
    names.forEach(function(name){ var m=models[name]; var c=el("div","ecard"); var r=el("div","er1");
      r.appendChild(el("span","ename",name)); r.appendChild(el("span","ptype",m.strategy)); if(m.provider) r.appendChild(el("span","badge",m.provider));
      var acts=el("div","eacts"); acts.appendChild(mkBtn("Edit","act",false,function(){ modelForm(name,m); }));
      acts.appendChild(mkBtn("Delete","act danger",false,function(){ confirmAction("Delete","Delete model <b>"+esc(name)+"</b>? Clients using it will get 404 until re-created.",true,function(){ deleteModel(name); }); }));
      r.appendChild(acts); c.appendChild(r); c.appendChild(el("div","edesc",strategySummary(m))); box.appendChild(c); });
  }
  function providerOptions(){ var ids=groupIds(); var opts=[]; if(ids.length!==1) opts.push({label:"(choose provider)",value:""}); ids.forEach(function(i){ opts.push({label:i,value:i}); }); return opts; }
  function modelForm(name, existing){
    var isNew=!existing; var fName, fStrat, fProv, dyn={};
    openForm(isNew?"Create model":("Edit model "+name), function(body){
      fName=fText("Model name","The virtual model id your client requests (e.g. fusion-coder, fast-glm).", isNew?"":name, true);
      if(!isNew){ fName.querySelector("input").disabled=true; }
      fStrat=fSelect("Strategy","single = pass to one model. failover = try a list in order. fusion = panel of experts → judge → synth. smart = a router picks cheap vs deep per request.", existing?existing.strategy:"fusion",
        [{label:"fusion",value:"fusion"},{label:"single",value:"single"},{label:"failover",value:"failover"},{label:"smart",value:"smart"}]);
      var ids=groupIds();
      fProv=fSelect("Provider group"+(ids.length===1?" (optional)":""),"Which provider serves this model. All its accounts share the same models, so failover stays consistent.", existing&&existing.provider?existing.provider:(ids.length===1?ids[0]:""), providerOptions());
      var host=el("div"); body.appendChild(fName); body.appendChild(fStrat); body.appendChild(fProv); body.appendChild(host);
      var provModels=[];
      // Refresh the provider catalog into the fields' suggestion lists IN PLACE.
      // NEVER rebuild on a refetch — destroying/recreating the fields would drop
      // focus, cursor position, and a half-typed tag in the add-box. Only the
      // upstream-model fields carry the provider catalog; smart's simple/fusion
      // routes suggest VIRTUAL model names, which don't change with the provider.
      function applySuggest(){ ["target","chain","panel","judge","synth","adv","router"].forEach(function(k){ if(dyn[k]&&dyn[k]._setSuggest) dyn[k]._setSuggest(provModels); }); }
      function rebuild(){ host.textContent=""; dyn={}; buildStrategyFields(host, fStrat._get(), existing); }
      function refetch(){ fetchProviderModels(fProv._get(), function(m){ provModels=m; applySuggest(); }); }
      fStrat.querySelector("select").onchange=rebuild;
      fProv.querySelector("select").onchange=refetch;
      rebuild();   // render immediately (no suggestions yet, never a blank form)
      refetch();   // then enrich the model fields with the provider's live catalog
      function buildStrategyFields(h, strat, ex){
        var up=provModels;      // real upstream models fetched from the provider's catalog
        var virt=modelNames();  // configured virtual model names (used by smart routes)
        if(strat==="single"){ dyn.target=fText("Target model","The one upstream model id this forwards to (e.g. glm-5.2). Pick from the provider's list.", ex&&ex.target, true, up); h.appendChild(dyn.target);
          addPromote(h, ex); addOverrides(h, ex); }
        else if(strat==="failover"){ dyn.chain=fTags("Chain","Upstream models tried IN ORDER; advance to the next on failure. Pick from the provider's list.", ex&&ex.chain, up); h.appendChild(dyn.chain);
          addPromote(h, ex); }
        else if(strat==="fusion"){
          dyn.panel=fTags("Panel (experts)","The models that each answer independently. Mix different model families for decorrelated views. Pick from the provider's list.", ex&&ex.panel, up); h.appendChild(dyn.panel);
          dyn.judge=fText("Judge","Ranks the experts' answers and picks the best (reliable structured output helps).", ex&&ex.judge, true, up); h.appendChild(dyn.judge);
          dyn.synth=fText("Synthesizer","Writes the final answer from the experts + judge.", ex&&ex.synth, true, up); h.appendChild(dyn.synth);
          dyn.adv=fText("Adversarial (optional)","One PANEL member that argues against the others to find flaws. Must be listed in the panel.", ex&&ex.adversarial, true, up); h.appendChild(dyn.adv);
          dyn.tool=fSelect("Tool mode","deliberate = experts discuss in prose, only the synth calls tools. bypass = straight to the synth with tools (faster, no deliberation).", ex?ex.tool_mode:"deliberate",[{label:"deliberate",value:"deliberate"},{label:"bypass",value:"bypass"}]); h.appendChild(dyn.tool);
          dyn.planOnly=fToggle("Full fusion only on planning turns","On: full panel on fresh instructions, synth-only on mid-loop tool-result continuations (cheaper). Off: full fusion every step.", ex?!!ex.fusion_planning_turn_only:false); h.appendChild(dyn.planOnly);
          // Web search grounding + its tuning (revealed only when enabled).
          var ws=(ex&&ex.web_search)||{};
          dyn.web=fToggle("Web search grounding","Run one web search before the panel and inject results as context. Needs TAVILY_API_KEY set on the server.", !!ws.enabled); h.appendChild(dyn.web);
          var wsub=subGroup();
          dyn.wsMax=fNum("Max results","How many search results to fetch (1–10). Default 3.", ws.max_results); wsub.appendChild(dyn.wsMax);
          dyn.wsTimeout=fNum("Search timeout (s)","Deadline for the search call (below 60). Default 20.", ws.timeout_s); wsub.appendChild(dyn.wsTimeout);
          dyn.wsCtx=fNum("Max context chars","Cap on injected context size. Default 4000.", ws.max_context_chars); wsub.appendChild(dyn.wsCtx);
          dyn.wsPrompt=fNum("Skip if prompt over (chars)","Skip grounding when the request is already this large, so context can't overflow a small-context member. Default 80000.", ws.max_prompt_chars); wsub.appendChild(dyn.wsPrompt);
          h.appendChild(wsub); bindReveal(dyn.web, wsub);
          // BinEval quality evaluation + its tuning (revealed only when enabled).
          var be=(ex&&ex.bineval)||{};
          dyn.bineval=fToggle("Quality evaluation (BinEval)","After the synth, score the answer on binary quality questions; results go into response headers (non-streaming only).", !!be.enabled); h.appendChild(dyn.bineval);
          var bsub=subGroup();
          dyn.beModel=fText("Evaluator model (optional)","Model that runs the evaluation. Blank = use the judge model.", be.model, true, up); bsub.appendChild(dyn.beModel);
          dyn.beThresh=fNum("Low-quality threshold","Overall score (0–1) below which the answer is flagged in the headers. Default 0.7.", be.threshold); bsub.appendChild(dyn.beThresh);
          dyn.beTimeout=fNum("Eval timeout (s)","Per-evaluation deadline. Blank = use the judge timeout.", be.timeout_s); bsub.appendChild(dyn.beTimeout);
          h.appendChild(bsub); bindReveal(dyn.bineval, bsub);
          addPromote(h, ex); addOverrides(h, ex);
          // Synth-only overrides (e.g. reasoning_effort → none): kept DISTINCT from
          // request_overrides, which the fusion strategy ignores. Without this control
          // a panel save would silently wipe synth_request_overrides (same round-trip
          // class as the web_search/bineval fix in v0.1.32) and re-open the synth leak.
          dyn.synthOverrides=fKV("Synth request overrides (optional)","Extra request-body fields sent upstream to the SYNTH stage only, e.g. reasoning_effort → none (stops the synth from leaking its reasoning). Panel & judge are unaffected.", (ex&&ex.synth_request_overrides)||{}); h.appendChild(dyn.synthOverrides);
        }
        else if(strat==="smart"){
          dyn.router=fText("Router model","A fast model that classifies each request as simple vs deep (needs reliable JSON). Pick from the provider's list.", ex&&ex.router, true, up); h.appendChild(dyn.router);
          dyn.def=fSelect("Default route","Used when the router is unsure or errors.", ex?ex.default:"simple",[{label:"simple",value:"simple"},{label:"fusion",value:"fusion"}]); h.appendChild(dyn.def);
          dyn.simple=fText("Simple route","Name of a single/failover model to use for cheap steps (a model from the Models list).", ex&&typeof ex.simple==="string"?ex.simple:"", true, virt); h.appendChild(dyn.simple);
          dyn.fusion=fText("Fusion route","Name of a fusion model to use for deep steps (must be in the same provider group).", ex&&typeof ex.fusion==="string"?ex.fusion:"", true, virt); h.appendChild(dyn.fusion);
        }
      }
      // Per-model override of the global promote_reasoning_to_content. Tri-state:
      // inherit (omit the key) / on / off — a plain toggle can't express "unset".
      function addPromote(h, ex){ var v=(ex&&ex.promote_reasoning_to_content); dyn.promote=fSelect("Promote reasoning to content","Normalize a reasoning-only reply so plain clients see the answer. inherit = use the global default.", v==null?"inherit":(v?"on":"off"),[{label:"inherit (global default)",value:"inherit"},{label:"on",value:"on"},{label:"off",value:"off"}]); h.appendChild(dyn.promote); }
      // Extra request-body fields merged into every upstream call for this model
      // (e.g. reasoning_effort → none). Core keys are protected server-side.
      function addOverrides(h, ex){ dyn.overrides=fKV("Request overrides (optional)","Extra request-body fields sent upstream for this model, e.g. reasoning_effort → none. Values are sent as strings.", (ex&&ex.request_overrides)||{}); h.appendChild(dyn.overrides); }
    }, function(){
      var nm=fName._get(); if(!nm){ formError("Model name is required."); return; }
      var strat=fStrat._get(); var obj={ strategy:strat }; var prov=fProv._get(); if(prov) obj.provider=prov;
      // Collect the per-model promote override (tri-state) + request_overrides onto obj.
      function applyCommon(o){ if(dyn.promote){ var pv=dyn.promote._get(); if(pv==="on") o.promote_reasoning_to_content=true; else if(pv==="off") o.promote_reasoning_to_content=false; }
        if(dyn.overrides){ var ov=dyn.overrides._get(); if(Object.keys(ov).length) o.request_overrides=ov; } }
      if(strat==="single"){ var t=dyn.target._get(); if(!t){ formError("Target model is required."); return; } obj.target=t; applyCommon(obj); }
      else if(strat==="failover"){ var ch=dyn.chain._get(); if(!ch.length){ formError("Add at least one model to the chain."); return; } obj.chain=ch; applyCommon(obj); }
      else if(strat==="fusion"){ var panel=dyn.panel._get(); if(panel.length<1){ formError("Add at least one panel member."); return; }
        var judge=dyn.judge._get(), synth=dyn.synth._get(); if(!judge||!synth){ formError("Judge and Synthesizer are required."); return; }
        obj.panel=panel; obj.judge=judge; obj.synth=synth; var adv=dyn.adv._get(); if(adv) obj.adversarial=adv;
        obj.tool_mode=dyn.tool._get(); if(dyn.planOnly._get()) obj.fusion_planning_turn_only=true;
        if(dyn.web._get()){ var ws={ enabled:true };
          var wm=dyn.wsMax._get(); if(wm!==undefined){ if(isNaN(wm)){ formError("Web search max results must be a number."); return; } ws.max_results=wm; }
          var wt=dyn.wsTimeout._get(); if(wt!==undefined){ if(isNaN(wt)){ formError("Web search timeout must be a number."); return; } ws.timeout_s=wt; }
          var wc=dyn.wsCtx._get(); if(wc!==undefined){ if(isNaN(wc)){ formError("Web search max context chars must be a number."); return; } ws.max_context_chars=wc; }
          var wp=dyn.wsPrompt._get(); if(wp!==undefined){ if(isNaN(wp)){ formError("Web search prompt cap must be a number."); return; } ws.max_prompt_chars=wp; }
          obj.web_search=ws; }
        if(dyn.bineval._get()){ var be={ enabled:true };
          var bm=dyn.beModel._get(); if(bm) be.model=bm;
          var bth=dyn.beThresh._get(); if(bth!==undefined){ if(isNaN(bth)){ formError("BinEval threshold must be a number."); return; } be.threshold=bth; }
          var bto=dyn.beTimeout._get(); if(bto!==undefined){ if(isNaN(bto)){ formError("BinEval timeout must be a number."); return; } be.timeout_s=bto; }
          if(existing&&existing.bineval&&existing.bineval.dimensions) be.dimensions=existing.bineval.dimensions; // preserve custom questions
          obj.bineval=be; }
        if(dyn.synthOverrides){ var so=dyn.synthOverrides._get(); if(Object.keys(so).length) obj.synth_request_overrides=so; }
        applyCommon(obj);
      }
      else if(strat==="smart"){ var router=dyn.router._get(); if(!router){ formError("Router model is required."); return; }
        obj.router=router; obj.default=dyn.def._get(); var s=dyn.simple._get(), fu=dyn.fusion._get();
        if(!s||!fu){ formError("Simple and Fusion route model names are required."); return; } obj.simple=s; obj.fusion=fu;
      }
      saveModel(nm, obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
  document.getElementById("add-model").onclick=function(){ modelForm(null,null); };

  // ---- Settings tab (global, non-fusion-specific config) ----
  function saveSettings(section, obj, done){ jsend("PUT","admin/config/settings/"+encodeURIComponent(section), obj).then(function(){ toast(section+" settings saved","ok"); reloadConfigSoon(); if(done)done(true); })
    .catch(function(e){ if(done)done(false,String(e.message||e)); }); }
  // Numbers that only apply at boot (bind/port, concurrency, timeouts) need a
  // restart; the settings card flags them and pairs saving with the Restart button.
  function settingsCard(title, desc, onEdit, needsRestart){ var c=el("div","ecard"); var r=el("div","er1");
    r.appendChild(el("span","ename",title)); if(needsRestart){ var b=el("span","badge","restart to apply"); b.style.color="var(--cooling)"; r.appendChild(b); }
    var acts=el("div","eacts"); acts.appendChild(mkBtn("Edit","act",false,onEdit)); r.appendChild(acts); c.appendChild(r);
    c.appendChild(el("div","edesc",desc)); return c; }
  function renderSettings(){ var box=document.getElementById("settings-editor"); box.textContent=""; if(!cfg) return;
    var sv=cfg.server||{}, up=cfg.upstream||{}, df=cfg.defaults||{};
    box.appendChild(settingsCard("Server",
      "Listen on "+esc(sv.bind||"127.0.0.1")+":"+(sv.port||8080)+(sv.auth_token_env?(" · client auth "+esc(sv.auth_token_env)):" · no client auth")+(sv.admin_token_env?(" · admin token "+esc(sv.admin_token_env)):""),
      function(){ serverForm(sv); }, true));
    box.appendChild(settingsCard("Upstream",
      "api mode "+esc(up.api_mode||"auto")+" · max concurrency "+(up.max_concurrency==null?4:up.max_concurrency)+" · request timeout "+(up.request_timeout_s==null?170:up.request_timeout_s)+"s",
      function(){ upstreamForm(up); }, true));
    box.appendChild(settingsCard("Fusion defaults",
      "panel timeout "+(df.panel_member_timeout_s==null?90:df.panel_member_timeout_s)+"s · judge "+(df.judge_timeout_s==null?60:df.judge_timeout_s)+"s · min panel success "+(df.min_panel_success==null?1:df.min_panel_success),
      function(){ defaultsForm(df); }, false));
  }
  function serverForm(sv){ var fBind,fPort,fAuth,fAdmin;
    openForm("Edit server settings", function(body){
      var note=el("div","hint"); note.style.margin="0 2px 14px"; note.textContent="Bind address and port only take effect after a restart (use the Restart button)."; body.appendChild(note);
      fBind=fText("Bind address","Interface to listen on. 127.0.0.1 = localhost only (recommended). 0.0.0.0 = all interfaces (needs a client auth token or FUSION_ALLOW_OPEN).", sv.bind||"127.0.0.1", true);
      fPort=fNum("Port","TCP port for the proxy + panel (1–65535). Default 8080.", sv.port==null?8080:sv.port);
      fAuth=fText("Client auth token env var (optional)","Env var holding the client bearer token for /v1 requests. Set it to require auth (needed for non-loopback bind). Blank = no client auth.", sv.auth_token_env||"", true);
      fAdmin=fText("Admin token env var (optional)","Env var holding a SEPARATE token for this panel + /admin API, so the widely-copied client token doesn't also grant config edits + restart. Blank = the admin surface reuses the client token (or is loopback-only when neither is set).", sv.admin_token_env||"", true);
      body.appendChild(fBind); body.appendChild(fPort); body.appendChild(fAuth); body.appendChild(fAdmin);
    }, function(){
      var port=fPort._get(); if(port===undefined||isNaN(port)){ formError("Port is required and must be a number."); return; }
      var bind=fBind._get(); if(!bind){ formError("Bind address is required."); return; }
      var obj={ bind:bind, port:port }; var a=fAuth._get(); if(a) obj.auth_token_env=a; var ad=fAdmin._get(); if(ad) obj.admin_token_env=ad;
      saveSettings("server", obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
  function upstreamForm(up){ var fMode,fConc,fTimeout,fCool,fRecheck,fPerDef;
    openForm("Edit upstream settings", function(body){
      var note=el("div","hint"); note.style.margin="0 2px 14px"; note.textContent="Concurrency and timeouts are read at boot — restart to apply. base_url / API keys live under Providers."; body.appendChild(note);
      fMode=fSelect("API mode","auto = detect per provider. openai = force the /v1 OpenAI shape. native = force the Ollama /api shape.", up.api_mode||"auto",[{label:"auto",value:"auto"},{label:"openai",value:"openai"},{label:"native",value:"native"}]);
      fConc=fNum("Max concurrency","Global cap on simultaneous upstream requests. Default 4.", up.max_concurrency==null?4:up.max_concurrency);
      fTimeout=fNum("Request timeout (s)","Per-request upstream deadline. Must be below ~182s (the Ollama Cloud ceiling). Default 170.", up.request_timeout_s==null?170:up.request_timeout_s);
      fCool=fNum("Connector cooldown (s)","How long a connector rests after a soft failure (rate-limit/5xx/timeout) before it is probed again. Default 60.", up.connector_cooldown_s==null?60:up.connector_cooldown_s);
      fRecheck=fNum("Connector down recheck (s)","How long a connector stays down after a HARD failure (auth/quota) before an automatic probe. 0 = never auto-probe (manual reset only). Default 900.", up.connector_down_recheck_s==null?900:up.connector_down_recheck_s);
      fPerDef=fNum("Per-model concurrency default (optional)","Default budget applied to each model's own gate. Blank = a model may use the full global budget.", up.per_model_concurrency_default);
      body.appendChild(fMode); body.appendChild(fConc); body.appendChild(fTimeout); body.appendChild(fCool); body.appendChild(fRecheck); body.appendChild(fPerDef);
    }, function(){
      var obj=Object.assign({},up); // preserve base_url/api_key_env/per_model_concurrency the form doesn't edit
      obj.api_mode=fMode._get();
      var conc=fConc._get(); if(conc===undefined||isNaN(conc)){ formError("Max concurrency is required and must be a number."); return; } obj.max_concurrency=conc;
      var to=fTimeout._get(); if(to===undefined||isNaN(to)){ formError("Request timeout is required and must be a number."); return; } obj.request_timeout_s=to;
      var cool=fCool._get(); if(cool===undefined||isNaN(cool)){ formError("Connector cooldown is required and must be a number."); return; } obj.connector_cooldown_s=cool;
      var rc=fRecheck._get(); if(rc===undefined||isNaN(rc)){ formError("Connector down recheck is required and must be a number."); return; } obj.connector_down_recheck_s=rc;
      var pd=fPerDef._get(); if(pd===undefined) delete obj.per_model_concurrency_default; else if(isNaN(pd)){ formError("Per-model concurrency default must be a number."); return; } else obj.per_model_concurrency_default=pd;
      saveSettings("upstream", obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
  function defaultsForm(df){ var fPanel,fJudge,fRouter,fMin,fPromote;
    openForm("Edit fusion defaults", function(body){
      fPanel=fNum("Panel member timeout (s)","Per-model deadline for one panel answer. Default 90.", df.panel_member_timeout_s==null?90:df.panel_member_timeout_s);
      fJudge=fNum("Judge timeout (s)","Deadline for the judge's one scoring call. Default 60.", df.judge_timeout_s==null?60:df.judge_timeout_s);
      fRouter=fNum("Router timeout (s)","Deadline for a smart model's routing call; on timeout it uses the default route. Default 30.", df.router_timeout_s==null?30:df.router_timeout_s);
      fMin=fNum("Min panel success","Fewest usable panel answers required to proceed to judge/synth. Permanently-gated members (403/404/410) relax this automatically. Default 1.", df.min_panel_success==null?1:df.min_panel_success);
      fPromote=fToggle("Promote reasoning to content","On: a reasoning-only reply (text in the reasoning field, empty content) is normalized so plain clients still see the answer. A fusion model can override this per-model.", df.promote_reasoning_to_content==null?true:!!df.promote_reasoning_to_content);
      body.appendChild(fPanel); body.appendChild(fJudge); body.appendChild(fRouter); body.appendChild(fMin); body.appendChild(fPromote);
    }, function(){
      var obj={};
      var p=fPanel._get(); if(p===undefined||isNaN(p)){ formError("Panel member timeout is required and must be a number."); return; } obj.panel_member_timeout_s=p;
      var j=fJudge._get(); if(j===undefined||isNaN(j)){ formError("Judge timeout is required and must be a number."); return; } obj.judge_timeout_s=j;
      var r=fRouter._get(); if(r===undefined||isNaN(r)){ formError("Router timeout is required and must be a number."); return; } obj.router_timeout_s=r;
      var m=fMin._get(); if(m===undefined||isNaN(m)){ formError("Min panel success is required and must be a number."); return; } obj.min_panel_success=m;
      obj.promote_reasoning_to_content=fPromote._get();
      saveSettings("defaults", obj, function(ok,err){ if(ok) closeForm(); else formError(err); });
    });
  }
  document.getElementById("restart-btn").onclick=function(){
    confirmAction("Restart", "Restart the proxy now to apply boot-only settings (bind, port, concurrency, timeouts)? In-flight requests are dropped and the service is unavailable for a second or two while the supervisor relaunches it.", true, function(){
      jsend("POST","admin/restart").then(function(){ toast("restarting — the panel will reconnect shortly","ok"); })
        .catch(function(e){ toast(String(e.message||e),"err"); });
    });
  };

  // --- tabs ---------------------------------------------------------------
  var TAB_ORDER=["monitor","providers","models","settings"];
  function switchTab(t){ activeTab=t;
    TAB_ORDER.forEach(function(x){ var on=x===t;
      document.getElementById("tab-"+x).className="tab"+(on?" on":""); });
    var btns=document.querySelectorAll(".tab-btn"); Array.prototype.forEach.call(btns,function(b){ var on=b.getAttribute("data-tab")===t; b.className="tab-btn"+(on?" on":""); b.setAttribute("aria-selected", on?"true":"false"); b.tabIndex=on?0:-1; });
    if((t==="providers"||t==="models"||t==="settings") && !cfg){ loadConfig().catch(function(e){ if(String(e.message)!=="auth") toast("could not load config: "+e.message,"err"); }); }
  }
  Array.prototype.forEach.call(document.querySelectorAll(".tab-btn"),function(b){
    b.onclick=function(){ switchTab(b.getAttribute("data-tab")); };
    b.addEventListener("keydown",function(e){ var i=TAB_ORDER.indexOf(b.getAttribute("data-tab")), j=-1;
      if(e.key==="ArrowRight"||e.key==="ArrowDown") j=(i+1)%TAB_ORDER.length;
      else if(e.key==="ArrowLeft"||e.key==="ArrowUp") j=(i-1+TAB_ORDER.length)%TAB_ORDER.length;
      else if(e.key==="Home") j=0; else if(e.key==="End") j=TAB_ORDER.length-1;
      if(j>=0){ e.preventDefault(); switchTab(TAB_ORDER[j]); document.getElementById("tabbtn-"+TAB_ORDER[j]).focus(); } });
  });

  // --- token bar + global keys -------------------------------------------
  document.getElementById("tokensave").onclick=function(){ setTok(document.getElementById("tokenin").value.trim()); showTokenBar(false); tick(); if(cfg===null&&activeTab!=="monitor") loadConfig(); };
  document.getElementById("tokenin").addEventListener("keydown",function(e){ if(e.key==="Enter") document.getElementById("tokensave").click(); });
  document.addEventListener("keydown",function(e){ if(e.key==="Escape"){ closeConfirm(); closeForm(); } });

  // --- poll loop (monitor) ------------------------------------------------
  function tick(){ jget("admin/providers").then(function(j){ last=j; since=Date.now(); document.getElementById("hb").classList.add("beat"); renderMonitor(j); setText(document.getElementById("updated"),"updated just now"); })
    .catch(function(e){ if(String(e.message)!=="auth"){ setText(document.getElementById("updated"),"disconnected — retrying"); document.getElementById("hb").classList.remove("beat"); } }); }
  tick(); setInterval(tick, POLL_MS);
  setInterval(function(){ if(last){ renderMonitor(last); setText(document.getElementById("updated"),"updated "+rel(since)); } }, 1000);
})();
</script>
</body>
</html>`;
