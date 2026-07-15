/**
 * Self-contained connector-panel HTML (no external requests — inline CSS + JS, so
 * it renders on localhost with no network). Polls `/admin/providers` and renders,
 * per provider group, a section with one card per account (state/reason/metrics)
 * plus manual controls (disable/enable/reset/make-active). Updates are applied
 * IN PLACE (a two-level keyed reconcile — no innerHTML teardown), so the panel
 * never flickers. When the proxy has a client auth token configured, the panel
 * prompts for it once, stores it in localStorage, and sends it as a Bearer header.
 *
 * The client script deliberately avoids template literals so this file needs no
 * backtick/`${` escaping.
 */
export const PANEL_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>llm-fusion · connectors</title>
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
  body{
    background:var(--bg); color:var(--fg); font-family:var(--sans);
    font-size:14px; line-height:1.45; -webkit-font-smoothing:antialiased;
    padding:28px 22px 60px; max-width:1180px; margin:0 auto;
  }
  .num{font-variant-numeric:tabular-nums}
  .mono{font-family:var(--mono)}
  a{color:var(--accent)}

  header.top{display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin-bottom:20px}
  .brand{display:flex; align-items:center; gap:10px; font-weight:650; font-size:16px; letter-spacing:.2px}
  .brand .logo{width:22px;height:22px;border-radius:7px;background:
    linear-gradient(135deg,var(--accent),#9b7bff); box-shadow:var(--shadow)}
  .spacer{flex:1}
  .pill{display:inline-flex; align-items:center; gap:7px; padding:5px 12px; border-radius:999px;
    font-size:12.5px; font-weight:600; border:1px solid var(--line-2); background:var(--panel)}
  .pill .dot{width:8px;height:8px;border-radius:50%}
  .pill.ok{color:var(--up)} .pill.ok .dot{background:var(--up)}
  .pill.warn{color:var(--cooling)} .pill.warn .dot{background:var(--cooling)}
  .pill.bad{color:var(--down)} .pill.bad .dot{background:var(--down)}
  .updated{color:var(--muted); font-size:12.5px; display:flex; align-items:center; gap:7px}
  .heartbeat{width:7px;height:7px;border-radius:50%;background:var(--accent);opacity:.9}
  .beat{animation:beat 2s ease-in-out infinite}
  @keyframes beat{0%,100%{opacity:.25;transform:scale(.85)}50%{opacity:1;transform:scale(1)}}

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
  .ptype{font-size:10.5px; color:var(--muted); border:1px solid var(--line-2); border-radius:6px;
    padding:1px 7px; text-transform:uppercase; letter-spacing:.5px}
  .phead .pmeta{color:var(--muted); font-size:12px}

  .grid{display:grid; grid-template-columns:repeat(auto-fill,minmax(340px,1fr)); gap:14px}
  .card{background:var(--panel); border:1px solid var(--line); border-radius:var(--radius);
    padding:16px 16px 14px; box-shadow:var(--shadow); position:relative;
    transition:transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    animation:rise .22s ease both}
  @keyframes rise{from{opacity:0;transform:translateY(6px)}to{opacity:1;transform:none}}
  .card.active{border-color:color-mix(in srgb,var(--accent) 55%, var(--line))}
  .card.active:before{content:"";position:absolute;inset:-1px;border-radius:var(--radius);
    box-shadow:0 0 0 1px var(--accent) inset, 0 0 22px var(--accent-bg);pointer-events:none;opacity:.55}
  .card .row1{display:flex; align-items:center; gap:9px; margin-bottom:8px}
  .sdot{width:11px;height:11px;border-radius:50%;flex:none;box-shadow:0 0 0 3px var(--panel), 0 0 0 4px transparent}
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
  .badge{font-size:10.5px; font-weight:700; letter-spacing:.6px; text-transform:uppercase;
    padding:2px 7px; border-radius:6px; border:1px solid var(--line-2); color:var(--muted)}
  .badge.active{color:var(--accent); border-color:color-mix(in srgb,var(--accent) 45%, var(--line-2)); background:var(--accent-bg)}
  .badge.pin{color:#9b7bff; border-color:color-mix(in srgb,#9b7bff 45%, var(--line-2))}
  .meta{color:var(--muted); font-size:12px; display:flex; align-items:center; gap:8px; margin:2px 0 10px}
  .host{font-family:var(--mono)}
  .reason{font-size:12.5px; padding:8px 10px; border-radius:9px; margin-bottom:11px; border:1px solid var(--line)}
  .reason.soft{background:var(--cooling-bg); border-color:color-mix(in srgb,var(--cooling) 30%, var(--line))}
  .reason.hard{background:var(--down-bg); border-color:color-mix(in srgb,var(--down) 30%, var(--line))}
  .reason .why{font-weight:650; text-transform:capitalize}
  .reason .err{color:var(--muted); word-break:break-word}
  .reason .cd{color:var(--muted)}
  .metrics{display:grid; grid-template-columns:repeat(3,1fr); gap:8px 12px; margin-bottom:12px}
  .metric .mk{color:var(--muted); font-size:10.5px; text-transform:uppercase; letter-spacing:.5px}
  .metric .mv{font-size:14px; font-weight:620; margin-top:1px}
  .actions{display:flex; gap:7px; flex-wrap:wrap; border-top:1px solid var(--line); padding-top:11px}
  button.act{font:inherit; font-size:12.5px; font-weight:600; cursor:pointer; color:var(--fg);
    background:var(--panel-2); border:1px solid var(--line-2); border-radius:8px; padding:5px 11px;
    transition:background .14s ease, border-color .14s ease, transform .05s ease}
  button.act:hover{border-color:var(--accent); background:var(--accent-bg)}
  button.act:active{transform:translateY(1px)}
  button.act:disabled{opacity:.4; cursor:not-allowed}
  button.act.danger:hover{border-color:var(--down); background:var(--down-bg); color:var(--down)}

  .empty{color:var(--muted); text-align:center; padding:60px 0}
  #tokenbar{display:none; align-items:center; gap:10px; background:var(--panel); border:1px solid var(--down);
    border-radius:12px; padding:12px 14px; margin-bottom:16px}
  #tokenbar input{flex:1; font:inherit; background:var(--panel-2); border:1px solid var(--line-2);
    border-radius:8px; padding:7px 10px; color:var(--fg)}
  #toasts{position:fixed; right:18px; bottom:18px; display:flex; flex-direction:column; gap:8px; z-index:20}
  .toast{background:var(--panel); border:1px solid var(--line-2); border-left:3px solid var(--accent);
    border-radius:9px; padding:10px 14px; box-shadow:var(--shadow); font-size:13px; animation:rise .18s ease both}
  .toast.err{border-left-color:var(--down)}

  /* Confirmation modal — every mutating action asks first. */
  .ovl{position:fixed; inset:0; background:rgba(4,7,11,.5); display:none; align-items:center;
    justify-content:center; z-index:30; backdrop-filter:blur(2px); padding:20px}
  .ovl.on{display:flex; animation:fade .12s ease both}
  @keyframes fade{from{opacity:0}to{opacity:1}}
  .modal{background:var(--panel); border:1px solid var(--line-2); border-radius:14px; box-shadow:var(--shadow);
    padding:20px 20px 16px; max-width:400px; width:100%; animation:rise .16s ease both}
  .modal h3{margin:0 0 8px; font-size:15px; display:flex; align-items:center; gap:9px}
  .modal h3 .idot{width:9px;height:9px;border-radius:50%; background:var(--cooling); flex:none}
  .modal h3.danger .idot{background:var(--down)}
  .modal p{margin:0 0 18px; color:var(--muted); font-size:13.5px; line-height:1.55}
  .modal p b{color:var(--fg); font-family:var(--mono); font-weight:600}
  .modal .row{display:flex; gap:8px; justify-content:flex-end}
  @media (prefers-reduced-motion: reduce){*{animation:none !important; transition:none !important}}
</style>
</head>
<body>
  <header class="top">
    <div class="brand"><span class="logo"></span> llm-fusion <span style="color:var(--muted);font-weight:500">· connectors</span></div>
    <span class="spacer"></span>
    <span id="overall" class="pill"><span class="dot"></span><span class="lbl">…</span></span>
    <span class="updated"><span id="hb" class="heartbeat"></span><span id="updated">connecting…</span></span>
  </header>

  <form id="tokenbar" onsubmit="return false;">
    <span>Auth required — paste the proxy token (<span class="mono">FUSION_PROXY_TOKEN</span>):</span>
    <input id="tokenin" type="password" placeholder="Bearer token" autocomplete="off" />
    <button class="act" id="tokensave" type="submit">Save</button>
  </form>

  <section id="summary" class="summary"></section>
  <section id="providers"></section>
  <div id="empty" class="empty" style="display:none">No providers reported.</div>

  <div id="ovl" class="ovl">
    <div class="modal" role="dialog" aria-modal="true">
      <h3 id="ovl-title"><span class="idot"></span><span id="ovl-title-txt">Confirm</span></h3>
      <p id="ovl-msg"></p>
      <div class="row">
        <button class="act" id="ovl-cancel" type="button">Cancel</button>
        <button class="act danger" id="ovl-ok" type="button">Confirm</button>
      </div>
    </div>
  </div>

  <div id="toasts"></div>

<script>
(function(){
  "use strict";
  var TOKEN_KEY = "fusion_panel_token";
  var POLL_MS = 3000;
  var last = null;      // last snapshot payload
  var busy = {};        // account ids with an in-flight action
  var groups = {};      // groupId -> { section, head, grid, cards } — reused across polls
  var sum = null;       // summary tile refs, built once
  var since = Date.now();

  function tok(){ try { return localStorage.getItem(TOKEN_KEY) || ""; } catch(e){ return ""; } }
  function setTok(v){ try { localStorage.setItem(TOKEN_KEY, v); } catch(e){} }
  function authHeaders(){ var h = {}; var t = tok(); if (t) h["authorization"] = "Bearer " + t; return h; }

  function el(tag, cls, text){ var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
  // In-place setters — only touch the DOM when the value actually changed.
  function setText(n, v){ if (n.textContent !== v) n.textContent = v; }
  function setClass(n, v){ if (n.className !== v) n.className = v; }
  function show(n, on){ var d = on ? "" : "none"; if (n.style.display !== d) n.style.display = d; }

  function fmtInt(n){ return (n == null ? 0 : n).toLocaleString("en-US"); }
  function pct(f, t){ if (!t) return "—"; return Math.round((1 - f / t) * 100) + "%"; }
  function rel(ts){
    if (!ts) return "—";
    var s = Math.max(0, Math.round((Date.now() - ts) / 1000));
    if (s < 60) return s + "s ago";
    var m = Math.floor(s / 60); if (m < 60) return m + "m ago";
    var h = Math.floor(m / 60); if (h < 24) return h + "h ago";
    return Math.floor(h / 24) + "d ago";
  }
  function secs(ms){ if (ms == null) return ""; var s = Math.ceil(ms / 1000); return s > 0 ? s + "s" : "0s"; }

  function toast(msg, isErr){
    var box = document.getElementById("toasts");
    var t = el("div", "toast" + (isErr ? " err" : ""), msg);
    box.appendChild(t);
    setTimeout(function(){ t.style.opacity = "0"; t.style.transform = "translateY(6px)"; }, 2600);
    setTimeout(function(){ if (t.parentNode) t.parentNode.removeChild(t); }, 3000);
  }

  function showTokenBar(s){ document.getElementById("tokenbar").style.display = s ? "flex" : "none"; }

  function get(){
    return fetch("admin/providers", { headers: authHeaders(), cache: "no-store" })
      .then(function(r){
        if (r.status === 401){ showTokenBar(true); throw new Error("auth"); }
        showTokenBar(false);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
  }

  function esc(s){
    return String(s).replace(/[&<>"]/g, function(c){ return { "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;" }[c]; });
  }

  // Every mutating action asks for confirmation first (destructive / data-changing).
  var pendingYes = null;
  function askConfirm(action, id, onYes){
    var danger = (action === "disable");
    var b = "<b>" + esc(id) + "</b>";
    var msgs = {
      disable: "Disable account " + b + "? It will stop serving requests until you enable it again.",
      enable: "Enable account " + b + " so it can serve requests again?",
      reset: "Reset account " + b + " back to healthy (clear its cooldown / failure state)?",
      pin: "Make account " + b + " the active one for its provider?",
      unpin: "Clear the pinned active account for this provider?"
    };
    var cap = action.charAt(0).toUpperCase() + action.slice(1);
    document.getElementById("ovl-title-txt").textContent = cap;
    document.getElementById("ovl-title").className = danger ? "danger" : "";
    document.getElementById("ovl-msg").innerHTML = msgs[action] || ("Confirm " + esc(action) + " on " + b + "?");
    var ok = document.getElementById("ovl-ok");
    ok.textContent = cap;
    ok.className = danger ? "act danger" : "act";
    pendingYes = onYes;
    document.getElementById("ovl").classList.add("on");
    // Focus Cancel (not Confirm) so a reflexive Enter/Space cancels rather than
    // confirming a destructive action — Confirm requires a deliberate click.
    document.getElementById("ovl-cancel").focus();
  }
  function closeConfirm(){ document.getElementById("ovl").classList.remove("on"); pendingYes = null; }

  function act(id, action){
    if (busy[id]) return;
    askConfirm(action, id, function(){ doAct(id, action); });
  }

  function doAct(id, action){
    if (busy[id]) return;
    busy[id] = true; render(last);
    var path = "admin/connectors/" + encodeURIComponent(id) + "/" + action;
    fetch(path, { method: "POST", headers: authHeaders() })
      .then(function(r){
        if (r.status === 401){ showTokenBar(true); throw new Error("auth required"); }
        return r.json().then(function(j){ if (!r.ok) throw new Error(j && j.error ? j.error : "HTTP " + r.status); return j; });
      })
      .then(function(j){ last = j; toast(id + " · " + action); render(last); })
      .catch(function(e){ toast(String(e.message || e), true); })
      .finally(function(){ delete busy[id]; render(last); });
  }

  function mkBtn(label, cls, disabled, fn){
    var b = el("button", cls, label); b.disabled = disabled; b.onclick = fn; return b;
  }

  function allAccounts(provs){
    var out = []; provs.forEach(function(p){ p.accounts.forEach(function(a){ out.push(a); }); }); return out;
  }
  function counts(items){
    var c = { up:0, cooling:0, down:0, off:0, reqs:0, fails:0 };
    items.forEach(function(a){
      if (a.state === "up") c.up++; else if (a.state === "cooling") c.cooling++;
      else if (a.state === "down") c.down++; else if (a.state === "off") c.off++;
      c.reqs += a.totalRequests; c.fails += a.totalFailures;
    });
    return c;
  }

  function overall(provs){
    var o = document.getElementById("overall");
    var c = counts(allAccounts(provs));
    setClass(o, "pill " + (c.down ? "bad" : c.cooling ? "warn" : "ok"));
    setText(o.querySelector(".lbl"), c.down ? (c.down + " down") : c.cooling ? (c.cooling + " cooling") : (c.up + " healthy"));
  }

  function ensureSummary(){
    if (sum) return;
    var box = document.getElementById("summary");
    sum = {};
    [["providers","providers"],["accounts","accounts"],["healthy","healthy"],["cooling","cooling"],["down / off","downoff"]].forEach(function(d){
      var s = el("div","stat"); s.appendChild(el("div","k",d[0]));
      var v = el("div","v num"); s.appendChild(v); box.appendChild(s); sum[d[1]] = v;
    });
    var s = el("div","stat"); s.appendChild(el("div","k","requests"));
    var v = el("div","v num"); var main = document.createTextNode(""); var sm = el("small");
    v.appendChild(main); v.appendChild(sm); s.appendChild(v); box.appendChild(s);
    sum.reqMain = main; sum.reqSub = sm;
  }

  function updateSummary(provs){
    ensureSummary();
    var items = allAccounts(provs); var c = counts(items);
    setText(sum.providers, String(provs.length));
    setText(sum.accounts, String(items.length));
    setText(sum.healthy, String(c.up));
    setText(sum.cooling, String(c.cooling));
    setText(sum.downoff, String(c.down + c.off));
    var r = fmtInt(c.reqs); if (sum.reqMain.nodeValue !== r) sum.reqMain.nodeValue = r;
    setText(sum.reqSub, " " + pct(c.fails, c.reqs) + " ok");
  }

  // Build an account card ONCE and return update(a) that mutates it in place.
  function buildCard(){
    var root = el("div","card");
    var r1 = el("div","row1");
    r1.appendChild(el("span","sdot"));
    var cid = el("span","cid"); r1.appendChild(cid);
    var state = el("span","state"); r1.appendChild(state);
    var sp = el("span"); sp.style.flex = "1"; r1.appendChild(sp);
    var activeBadge = el("span","badge active","active"); r1.appendChild(activeBadge);
    var pinBadge = el("span","badge pin","pinned"); r1.appendChild(pinBadge);
    root.appendChild(r1);

    var meta = el("div","meta");
    var host = el("span","host mono"); var nokey = el("span","badge","no key");
    meta.appendChild(host); meta.appendChild(nokey);
    root.appendChild(meta);

    var reason = el("div","reason");
    var why = el("span","why"); var sep = document.createTextNode(""); var errEl = el("span","err"); var cd = el("div","cd");
    reason.appendChild(why); reason.appendChild(sep); reason.appendChild(errEl); reason.appendChild(cd);
    root.appendChild(reason);

    var metrics = el("div","metrics");
    function metric(k){ var m = el("div","metric"); m.appendChild(el("div","mk",k)); var v = el("div","mv num"); m.appendChild(v); metrics.appendChild(m); return v; }
    var mReq = metric("requests"), mFail = metric("failures"), mSucc = metric("success"),
        mOk = metric("last ok"), mBad = metric("last fail"), mLat = metric("latency");
    root.appendChild(metrics);

    var actions = el("div","actions"); root.appendChild(actions);
    var actKey = "";

    function update(a){
      setClass(root, "card s-" + a.state + (a.active ? " active" : ""));
      setText(cid, a.id);
      setText(state, a.state);
      show(activeBadge, !!a.active);
      show(pinBadge, !!a.pinned);
      setText(host, a.host);
      show(nokey, !a.hasKey);

      var isReason = (a.state === "cooling" || a.state === "down" || a.state === "off");
      show(reason, isReason);
      if (isReason){
        var hard = a.state === "down";
        setClass(reason, "reason " + (hard ? "hard" : a.state === "off" ? "" : "soft"));
        setText(why, a.reason || (a.state === "off" ? "manual" : a.state));
        var hasErr = !!a.lastError;
        show(errEl, hasErr);
        var s = hasErr ? " · " : ""; if (sep.nodeValue !== s) sep.nodeValue = s;
        if (hasErr) setText(errEl, a.lastError);
        if (a.cooldownRemainingMs != null && a.cooldownRemainingMs > 0) setText(cd, (hard ? "recheck in " : "probe in ") + secs(a.cooldownRemainingMs));
        else if (a.state !== "off") setText(cd, "probing on next request");
        else setText(cd, "");
      }

      setText(mReq, fmtInt(a.totalRequests));
      setText(mFail, fmtInt(a.totalFailures));
      setText(mSucc, pct(a.totalFailures, a.totalRequests));
      setText(mOk, rel(a.lastSuccessAt));
      setText(mBad, rel(a.lastFailureAt));
      setText(mLat, a.lastLatencyMs != null ? a.lastLatencyMs + "ms" : "—");

      var b = !!busy[a.id];
      var key = a.state + "|" + (a.active ? 1 : 0) + "|" + (a.pinned ? 1 : 0) + "|" + (b ? 1 : 0);
      if (key !== actKey){
        actKey = key;
        actions.textContent = "";
        if (a.state === "off") actions.appendChild(mkBtn("Enable","act",b,function(){ act(a.id,"enable"); }));
        else actions.appendChild(mkBtn("Disable","act danger",b,function(){ act(a.id,"disable"); }));
        if (a.state === "down" || a.state === "cooling") actions.appendChild(mkBtn("Reset","act",b,function(){ act(a.id,"reset"); }));
        if (!a.active && a.state !== "off") actions.appendChild(mkBtn("Make active","act",b,function(){ act(a.id,"pin"); }));
        if (a.pinned) actions.appendChild(mkBtn("Unpin","act",b,function(){ act(a.id,"unpin"); }));
      }
    }

    return { root: root, update: update };
  }

  function buildGroup(){
    var section = el("div","provider");
    var head = el("div","phead");
    head.appendChild(el("span","pdot"));
    var pname = el("span","pname"); head.appendChild(pname);
    var ptype = el("span","ptype"); head.appendChild(ptype);
    var pmeta = el("span","pmeta"); head.appendChild(pmeta);
    section.appendChild(head);
    var grid = el("div","grid"); section.appendChild(grid);
    return { section: section, head: head, pname: pname, ptype: ptype, pmeta: pmeta, grid: grid, cards: {} };
  }

  function updateGroupHead(g, p){
    var c = counts(p.accounts);
    setClass(g.head, "phead " + (c.down ? "g-bad" : c.cooling ? "g-warn" : "g-ok"));
    setText(g.pname, p.id);
    setText(g.ptype, p.type);
    var active = p.activeId ? ("active: " + p.activeId) : "no active account";
    setText(g.pmeta, p.accounts.length + " account" + (p.accounts.length === 1 ? "" : "s") + " · " + active);
  }

  function render(payload){
    if (!payload) return;
    var provs = payload.providers || [];
    show(document.getElementById("empty"), provs.length === 0);
    overall(provs);
    updateSummary(provs);

    var container = document.getElementById("providers");
    var seenG = {};
    provs.forEach(function(p){
      seenG[p.id] = true;
      var g = groups[p.id];
      if (!g){ g = buildGroup(); groups[p.id] = g; container.appendChild(g.section); }
      updateGroupHead(g, p);
      var seenA = {};
      p.accounts.forEach(function(a){
        seenA[a.id] = true;
        var card = g.cards[a.id];
        if (!card){ card = buildCard(); g.cards[a.id] = card; g.grid.appendChild(card.root); }
        card.update(a);
      });
      Object.keys(g.cards).forEach(function(id){
        if (!seenA[id]){ if (g.cards[id].root.parentNode) g.grid.removeChild(g.cards[id].root); delete g.cards[id]; }
      });
    });
    Object.keys(groups).forEach(function(id){
      if (!seenG[id]){ if (groups[id].section.parentNode) container.removeChild(groups[id].section); delete groups[id]; }
    });
  }

  function tick(){
    get().then(function(j){
      last = j; since = Date.now();
      document.getElementById("hb").classList.add("beat");
      render(j);
      setText(document.getElementById("updated"), "updated just now");
    }).catch(function(e){
      if (String(e.message) !== "auth"){
        setText(document.getElementById("updated"), "disconnected — retrying");
        document.getElementById("hb").classList.remove("beat");
      }
    });
  }

  document.getElementById("tokensave").onclick = function(){
    setTok(document.getElementById("tokenin").value.trim()); showTokenBar(false); tick();
  };
  document.getElementById("tokenin").addEventListener("keydown", function(e){
    if (e.key === "Enter") document.getElementById("tokensave").click();
  });

  document.getElementById("ovl-cancel").onclick = closeConfirm;
  document.getElementById("ovl-ok").onclick = function(){ var f = pendingYes; closeConfirm(); if (f) f(); };
  document.getElementById("ovl").onclick = function(e){ if (e.target === this) closeConfirm(); };
  document.addEventListener("keydown", function(e){ if (e.key === "Escape") closeConfirm(); });

  tick();
  setInterval(tick, POLL_MS);
  // 1 s in-place refresh keeps relative times + cooldown countdowns live between
  // network polls — no fetch, no flicker (render() only writes changed values).
  setInterval(function(){ if (last){ render(last); setText(document.getElementById("updated"), "updated " + rel(since)); } }, 1000);
})();
</script>
</body>
</html>`;
