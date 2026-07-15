/**
 * Self-contained connector-panel HTML (no external requests — inline CSS + JS,
 * so it renders on localhost with no network). Polls `/admin/connectors`, renders
 * a summary + one card per connector with live state/reason/metrics, and offers
 * manual controls (disable/enable/reset/make-active). When the proxy has a client
 * auth token configured, the panel prompts for it once and stores it in
 * localStorage, sending it as a Bearer header on the data/action calls.
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

  .summary{display:grid; grid-template-columns:repeat(auto-fit,minmax(120px,1fr)); gap:10px; margin-bottom:22px}
  .stat{background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:12px 14px}
  .stat .k{color:var(--muted); font-size:11.5px; text-transform:uppercase; letter-spacing:.6px}
  .stat .v{font-size:22px; font-weight:680; margin-top:3px}
  .stat .v small{font-size:13px; color:var(--muted); font-weight:500}

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
  .meta .prov{font-weight:600; color:var(--fg); opacity:.8}
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
  <section id="grid" class="grid"></section>
  <div id="empty" class="empty" style="display:none">No connectors reported.</div>
  <div id="toasts"></div>

<script>
(function(){
  "use strict";
  var TOKEN_KEY = "fusion_panel_token";
  var POLL_MS = 3000;
  var last = null;      // last snapshot payload
  var busy = {};        // connector ids with an in-flight action

  function tok(){ try { return localStorage.getItem(TOKEN_KEY) || ""; } catch(e){ return ""; } }
  function setTok(v){ try { localStorage.setItem(TOKEN_KEY, v); } catch(e){} }
  function authHeaders(){ var h = {}; var t = tok(); if (t) h["authorization"] = "Bearer " + t; return h; }

  function el(tag, cls, text){ var e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; }
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

  function showTokenBar(show){ document.getElementById("tokenbar").style.display = show ? "flex" : "none"; }

  function get(){
    return fetch("admin/connectors", { headers: authHeaders(), cache: "no-store" })
      .then(function(r){
        if (r.status === 401){ showTokenBar(true); throw new Error("auth"); }
        showTokenBar(false);
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      });
  }

  function act(id, action){
    if (busy[id]) return;
    busy[id] = true; render(last);
    var path = action === "unpin" ? "admin/unpin" : "admin/connectors/" + encodeURIComponent(id) + "/" + action;
    fetch(path, { method: "POST", headers: authHeaders() })
      .then(function(r){
        if (r.status === 401){ showTokenBar(true); throw new Error("auth required"); }
        return r.json().then(function(j){ if (!r.ok) throw new Error(j && j.error ? j.error : "HTTP " + r.status); return j; });
      })
      .then(function(j){ last = j; toast((action === "unpin" ? "unpinned" : id + " · " + action)); render(last); })
      .catch(function(e){ toast(String(e.message || e), true); })
      .finally(function(){ delete busy[id]; render(last); });
  }

  function overall(items){
    var o = document.getElementById("overall"); var lbl = o.querySelector(".lbl");
    var down = items.filter(function(c){ return c.state === "down"; }).length;
    var cool = items.filter(function(c){ return c.state === "cooling"; }).length;
    var upN = items.filter(function(c){ return c.state === "up"; }).length;
    o.className = "pill " + (down ? "bad" : cool ? "warn" : "ok");
    lbl.textContent = down ? (down + " down") : cool ? (cool + " cooling") : (upN + " healthy");
  }

  function summary(items, activeId){
    var box = document.getElementById("summary"); box.innerHTML = "";
    var upN = items.filter(function(c){ return c.state === "up"; }).length;
    var cool = items.filter(function(c){ return c.state === "cooling"; }).length;
    var downOff = items.filter(function(c){ return c.state === "down" || c.state === "off"; }).length;
    var reqs = items.reduce(function(a, c){ return a + c.totalRequests; }, 0);
    var fails = items.reduce(function(a, c){ return a + c.totalFailures; }, 0);
    var active = items.filter(function(c){ return c.id === activeId; })[0];
    function stat(k, v, sub){
      var s = el("div", "stat"); s.appendChild(el("div", "k", k));
      var val = el("div", "v num"); val.textContent = v;
      if (sub){ var sm = el("small"); sm.textContent = " " + sub; val.appendChild(sm); }
      s.appendChild(val); return s;
    }
    box.appendChild(stat("connectors", String(items.length)));
    box.appendChild(stat("healthy", String(upN)));
    box.appendChild(stat("cooling", String(cool)));
    box.appendChild(stat("down / off", String(downOff)));
    box.appendChild(stat("active", active ? active.id : "—"));
    box.appendChild(stat("requests", fmtInt(reqs), pct(fails, reqs) + " ok"));
  }

  function metric(k, v){
    var m = el("div", "metric"); m.appendChild(el("div", "mk", k));
    m.appendChild(el("div", "mv num", v)); return m;
  }

  function card(c){
    var wrap = el("div", "card s-" + c.state + (c.active ? " active" : ""));
    var r1 = el("div", "row1");
    r1.appendChild(el("span", "sdot"));
    r1.appendChild(el("span", "cid", c.id));
    r1.appendChild(el("span", "state", c.state));
    var sp = el("span"); sp.style.flex = "1"; r1.appendChild(sp);
    if (c.active) r1.appendChild(el("span", "badge active", "active"));
    if (c.pinned) r1.appendChild(el("span", "badge pin", "pinned"));
    wrap.appendChild(r1);

    var meta = el("div", "meta");
    meta.appendChild(el("span", "prov", c.provider));
    meta.appendChild(el("span", "host mono", c.host));
    if (!c.hasKey) meta.appendChild(el("span", "badge", "no key"));
    wrap.appendChild(meta);

    if (c.state === "cooling" || c.state === "down" || c.state === "off"){
      var hard = c.state === "down";
      var box = el("div", "reason " + (hard ? "hard" : c.state === "off" ? "" : "soft"));
      var why = el("span", "why", c.reason || (c.state === "off" ? "manual" : c.state));
      box.appendChild(why);
      if (c.lastError){ box.appendChild(document.createTextNode(" · ")); box.appendChild(el("span", "err", c.lastError)); }
      if (c.cooldownRemainingMs != null && c.cooldownRemainingMs > 0){
        box.appendChild(el("div", "cd", (hard ? "recheck in " : "probe in ") + secs(c.cooldownRemainingMs)));
      } else if (c.state !== "off"){
        box.appendChild(el("div", "cd", "probing on next request"));
      }
      wrap.appendChild(box);
    }

    var m = el("div", "metrics");
    m.appendChild(metric("requests", fmtInt(c.totalRequests)));
    m.appendChild(metric("failures", fmtInt(c.totalFailures)));
    m.appendChild(metric("success", pct(c.totalFailures, c.totalRequests)));
    m.appendChild(metric("last ok", rel(c.lastSuccessAt)));
    m.appendChild(metric("last fail", rel(c.lastFailureAt)));
    m.appendChild(metric("latency", c.lastLatencyMs != null ? c.lastLatencyMs + "ms" : "—"));
    wrap.appendChild(m);

    var acts = el("div", "actions");
    var b = !!busy[c.id];
    if (c.state === "off"){
      acts.appendChild(mkBtn("Enable", "act", b, function(){ act(c.id, "enable"); }));
    } else {
      acts.appendChild(mkBtn("Disable", "act danger", b, function(){ act(c.id, "disable"); }));
    }
    if (c.state === "down" || c.state === "cooling"){
      acts.appendChild(mkBtn("Reset", "act", b, function(){ act(c.id, "reset"); }));
    }
    if (!c.active && c.state !== "off"){
      acts.appendChild(mkBtn("Make active", "act", b, function(){ act(c.id, "pin"); }));
    }
    if (c.pinned){
      acts.appendChild(mkBtn("Unpin", "act", b, function(){ act(c.id, "unpin"); }));
    }
    wrap.appendChild(acts);
    return wrap;
  }

  function mkBtn(label, cls, disabled, fn){
    var btn = el("button", cls, label); btn.disabled = disabled; btn.onclick = fn; return btn;
  }

  function render(payload){
    if (!payload) return;
    var items = payload.connectors || [];
    document.getElementById("empty").style.display = items.length ? "none" : "block";
    overall(items);
    summary(items, payload.activeId);
    var grid = document.getElementById("grid"); grid.innerHTML = "";
    items.forEach(function(c){ grid.appendChild(card(c)); });
  }

  function tick(){
    get().then(function(j){
      last = j;
      document.getElementById("updated").textContent = "updated just now";
      document.getElementById("hb").classList.add("beat");
      render(j);
    }).catch(function(e){
      if (String(e.message) !== "auth"){
        document.getElementById("updated").textContent = "disconnected — retrying";
        document.getElementById("hb").classList.remove("beat");
      }
    });
  }

  // Re-label "updated" as time passes (without a fetch), and refresh cooldown countdowns.
  var since = Date.now();
  function relabel(){
    if (last){ document.getElementById("updated").textContent = "updated " + rel(since); }
  }

  document.getElementById("tokensave").onclick = function(){
    setTok(document.getElementById("tokenin").value.trim());
    showTokenBar(false); tick();
  };
  document.getElementById("tokenin").addEventListener("keydown", function(e){
    if (e.key === "Enter") document.getElementById("tokensave").click();
  });

  function loop(){ since = Date.now(); tick(); }
  loop();
  setInterval(loop, POLL_MS);
  setInterval(relabel, 1000);
})();
</script>
</body>
</html>`;
