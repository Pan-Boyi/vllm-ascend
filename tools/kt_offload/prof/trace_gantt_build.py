#!/usr/bin/env python3
"""Render the two trace slices into one self-contained HTML <section>.

Input is what ``trace_slice.py`` wrote into
``build_artifacts/moe_offload_ref/trace_slices/`` -- the slices are the tracked
artefact, this script is only the pen. Output is a fragment: no
``<html>/<head>/<body>``, one ``<section class="tgslice">`` carrying its own
prefixed ``<style>`` and an IIFE, with both JSONs inlined (no ``fetch``, no
external library -- the host page's CSP only allows a few script origins and
this needs none of them).

Colour comes entirely from the host page's CSS variables; this file defines no
palette of its own. It uses, and will render wrong without:

    --ink --ink-2 --ink-3   text
    --panel                 card surface
    --rule --rule-2         hairlines / neutral fill
    --teal --orange --violet  the three (CVD-validated) hues

Six task classes on three hues, so colour is never the only channel: MEMCPY is
half-height teal against full-height teal compute, the host-func node is a 2.2px
tick, and every block names itself on hover. There is also a numbers table.

    trace_gantt_build.py serial.json side.json -o trace_gantt_section.html
"""
import argparse
import json


TEMPLATE = r'''<section class="tgslice" aria-labelledby="tgslice-h">
<style>
.tgslice{--tg-lane:30px;--tg-gap:6px;margin:2.5rem 0}
.tgslice h3{margin:0 0 .35em;color:var(--ink)}
.tgslice .tg-lede{margin:0 0 1.1em;color:var(--ink-2);max-width:62ch}
.tgslice .tg-lede code{color:var(--ink)}
.tgslice .tg-warn{margin:.8em 0 1.2em;padding:.6em .8em;border-left:2px solid var(--rule-2);color:var(--ink-3);font-size:.86em;max-width:62ch}
.tgslice .tg-ctl{display:flex;flex-wrap:wrap;gap:.45em;align-items:center;margin-bottom:.9em}
.tgslice .tg-ctl button{font:inherit;font-size:.82em;line-height:1;padding:.45em .7em;color:var(--ink-2);background:var(--panel);border:1px solid var(--rule);border-radius:5px;cursor:pointer}
.tgslice .tg-ctl button:hover{color:var(--ink);border-color:var(--rule-2)}
.tgslice .tg-ctl button[aria-pressed=true]{color:var(--ink);border-color:var(--ink-3);background:transparent}
.tgslice .tg-hint{font-size:.78em;color:var(--ink-3)}
.tgslice .tg-legend{display:flex;flex-wrap:wrap;gap:.35em 1.1em;list-style:none;margin:0 0 1.1em;padding:0;font-size:.8em;color:var(--ink-2)}
.tgslice .tg-legend li{display:flex;align-items:center;gap:.45em}
.tgslice .tg-legend i{display:inline-block;width:16px;height:11px;border-radius:2px;flex:none}
.tgslice .tg-legend i.k4{background:var(--teal)}
.tgslice .tg-legend i.k3{background:var(--teal);opacity:.42;height:6px}
.tgslice .tg-legend i.k1{background:var(--orange)}
.tgslice .tg-legend i.k0{background:var(--orange);width:3px;height:8px;align-self:flex-start;margin-top:2px}
.tgslice .tg-legend i.k2{background:var(--violet)}
.tgslice .tg-legend i.k5{background:var(--rule-2)}
.tgslice .tg-arm{margin:0 0 1.6em;padding:.85em .95em .7em;background:var(--panel);border:1px solid var(--rule);border-radius:8px}
.tgslice .tg-arm h4{margin:0 0 .15em;font-size:.95em;color:var(--ink)}
.tgslice .tg-arm .tg-meta{margin:0 0 .8em;font-size:.78em;color:var(--ink-3)}
.tgslice .tg-lane{display:grid;grid-template-columns:minmax(120px,168px) 1fr;gap:var(--tg-gap);align-items:center;margin-bottom:5px}
.tgslice .tg-lane .tg-lbl{font-size:.74em;line-height:1.3;color:var(--ink-2);text-align:right;overflow:hidden}
.tgslice .tg-lane .tg-lbl b{display:block;color:var(--ink);font-weight:600}
.tgslice .tg-lane .tg-lbl span{color:var(--ink-3)}
.tgslice .tg-plot{position:relative;min-width:0}
.tgslice svg.tg-svg{display:block;width:100%;height:var(--tg-lane);cursor:crosshair;touch-action:none;overflow:hidden}
.tgslice svg.tg-svg rect.b{shape-rendering:crispEdges}
.tgslice .c4{fill:var(--teal)}
.tgslice .c3{fill:var(--teal);opacity:.42}
.tgslice .c1{fill:var(--orange)}
.tgslice .c0{fill:var(--orange)}
.tgslice .c2{fill:var(--violet)}
.tgslice .c5{fill:var(--rule-2)}
.tgslice .tg-bg{fill:var(--rule);opacity:.28}
.tgslice .tg-stepline{stroke:var(--ink-3);stroke-width:1;stroke-dasharray:2 3;opacity:.7}
.tgslice .tg-brush{fill:var(--ink-3);opacity:.16;pointer-events:none}
.tgslice .tg-axis{display:grid;grid-template-columns:minmax(120px,168px) 1fr;gap:var(--tg-gap);margin-top:2px}
.tgslice .tg-axis .tg-ticks{position:relative;height:1.35em;font-size:.7em;color:var(--ink-3);min-width:0}
.tgslice .tg-axis .tg-ticks i{position:absolute;top:0;transform:translateX(-50%);font-style:normal;white-space:nowrap}
.tgslice .tg-tip{position:fixed;z-index:60;pointer-events:none;max-width:380px;padding:.45em .6em;font-size:.74em;line-height:1.45;color:var(--ink);background:var(--panel);border:1px solid var(--rule-2);border-radius:5px;box-shadow:0 2px 10px rgba(0,0,0,.18)}
.tgslice .tg-tip b{font-weight:600;overflow-wrap:anywhere}
.tgslice .tg-tip em{font-style:normal;color:var(--ink-3)}
.tgslice .tg-tip[hidden]{display:none}
.tgslice .tg-tbl{width:100%;border-collapse:collapse;font-size:.78em;margin-top:.4em}
.tgslice .tg-tbl caption{text-align:left;color:var(--ink-3);font-size:.95em;padding-bottom:.5em}
.tgslice .tg-tbl th,.tgslice .tg-tbl td{padding:.34em .5em;border-bottom:1px solid var(--rule);text-align:right;color:var(--ink-2)}
.tgslice .tg-tbl th:first-child,.tgslice .tg-tbl td:first-child{text-align:left}
.tgslice .tg-tbl thead th{color:var(--ink-3);font-weight:500;white-space:nowrap}
.tgslice .tg-tbl td.ok{color:var(--ink)}
@media(max-width:560px){
 .tgslice .tg-lane,.tgslice .tg-axis{grid-template-columns:1fr}
 .tgslice .tg-lane .tg-lbl{text-align:left}
}
</style>

<h3 id="tgslice-h">接缝的形状：串行臂 vs 侧流臂，各 2 个稳态 decode step</h3>
<p class="tg-lede">两块图各自来自一份 profiler trace 的设备侧任务表（<code>task_time.csv</code>），
只取稳态 decode 中段连续 2 个 step。上块是<b>串行臂</b>（侧流关）——每层的 <code>NOTIFY_WAIT</code> 直接压在主流上；
下块是<b>侧流臂</b>（侧流开、P1a 关）——同一段等待挪到了副流，主流换上 <code>DAVID_EVENT_WAIT</code>。
两块用<b>同一个 µs/px 比例</b>，所以块宽可以直接比；但<b>横轴各归各的零点</b>。</p>
<p class="tg-warn">⚠ trace 的时间戳是 <code>CLOCK_REALTIME</code>，两发相隔约 8 小时 ⇒
<b>两臂不能叠在同一条绝对时间轴上</b>。这里每臂的 t=0 是它自己切片首个 <code>MODEL_EXECUTE</code>，只比形状、不比位置。</p>

<div class="tg-ctl" role="group" aria-label="缩放">
  <button type="button" data-view="all" aria-pressed="true">两步全览</button>
  <button type="button" data-view="0" aria-pressed="false">只看第 1 步</button>
  <button type="button" data-view="1" aria-pressed="false">只看第 2 步</button>
  <span class="tg-hint">在泳道上拖动可框选放大（两臂同步同倍率），双击复位</span>
</div>

<ul class="tg-legend">
  <li><i class="k4"></i>计算 AI_CORE / MIX_AIC / AI_VECTOR_CORE</li>
  <li><i class="k3"></i>MEMCPY_ASYNC（半高）</li>
  <li><i class="k1"></i>NOTIFY_WAIT</li>
  <li><i class="k0"></i>HOSTFUNC_CALLBACK（0.007 µs，画成竖标）</li>
  <li><i class="k2"></i>*_EVENT_WAIT / MEM_WAIT_VALUE</li>
  <li><i class="k5"></i>其它</li>
</ul>

<div class="tg-arms"></div>

<table class="tg-tbl">
  <caption>切片读数 vs 已内联的整发聚合读数（同一 <code>task_time.csv</code>，口径见脚注）</caption>
  <thead><tr><th>读数</th><th>已内联（整发）</th><th>本切片（2 step）</th><th>差</th></tr></thead>
  <tbody class="tg-tbody"></tbody>
</table>

<div class="tg-tip" hidden></div>

<script>
(function(){
  var ROOT = document.currentScript.closest('section.tgslice');
  var DATA = [/*__DATA__*/];
  var CHECKS = /*__CHECKS__*/;
  // TOP is a marker gutter: the 0.007 us host-func node is drawn as a tick up
  // there instead of on the bar, because it starts at exactly the same ts as the
  // NOTIFY_WAIT it triggers and is the same colour -- on the bar it is invisible.
  var LANE = 30, TOP = 4, BH = LANE - TOP - 2, MINW = 0.7;
  var CLSN = ['HOSTFUNC_CALLBACK','NOTIFY_WAIT','EVENT_WAIT','MEMCPY_ASYNC','计算','其它'];
  var ROLE = {main:'主流', side:'副流', frame:'框架流', aux:'辅流'};
  var SVGNS = 'http://www.w3.org/2000/svg';
  var tip = ROOT.querySelector('.tg-tip');
  var armsBox = ROOT.querySelector('.tg-arms');
  var view = null;          // {span} shared, per-arm t0 in a.t0
  var arms = [];

  function fmt(us){
    if (us >= 1000) return (us/1000).toFixed(us >= 10000 ? 1 : 2) + ' ms';
    if (us >= 1) return us.toFixed(2) + ' µs';
    return us.toFixed(3) + ' µs';
  }
  function el(tag, cls, txt){ var n=document.createElement(tag); if(cls)n.className=cls; if(txt!=null)n.textContent=txt; return n; }

  // ---- build DOM ---------------------------------------------------------
  DATA.forEach(function(d, ai){
    var box = el('div','tg-arm');
    box.appendChild(el('h4', null, d.title));
    var meta = el('p','tg-meta');
    meta.textContent = d.run + ' · step #' + d.step_index.join(', #') +
      ' · 切片 ' + (d.span_us/1000).toFixed(1) + ' ms · ' + d.tasks.length + ' 个块' +
      '（碎任务 <0.5 µs 已合并 ' + d.merged_total.k + ' 条，合计 ' + d.merged_total.dur_us.toFixed(1) + ' µs）';
    box.appendChild(meta);

    var lanes = [];
    d.streams.forEach(function(s, si){
      var row = el('div','tg-lane');
      var lbl = el('div','tg-lbl');
      var b = el('b', null, (ROLE[s.role]||s.role) + ' · stream ' + s.id);
      lbl.appendChild(b);
      var tot = {};
      d.tasks.forEach(function(t){ if(t.s===si){ var c=d.name_class[t.n]; tot[c]=(tot[c]||0)+t.d; } });
      var parts = Object.keys(tot).map(Number).sort(function(x,y){return tot[y]-tot[x];})
        .filter(function(c){ return tot[c] >= 50; })
        .slice(0,3).map(function(c){ return CLSN[c]+' '+(tot[c]/1000).toFixed(1)+'ms'; });
      lbl.appendChild(el('span', null, parts.join(' · ')));
      row.appendChild(lbl);
      var plot = el('div','tg-plot');
      var svg = document.createElementNS(SVGNS,'svg');
      svg.setAttribute('class','tg-svg');
      svg.setAttribute('preserveAspectRatio','none');
      svg.dataset.arm = ai; svg.dataset.stream = si;
      plot.appendChild(svg); row.appendChild(plot); box.appendChild(row);
      lanes.push(svg);
    });

    var ax = el('div','tg-axis');
    ax.appendChild(el('div','tg-lbl'));
    var ticks = el('div','tg-ticks'); ax.appendChild(ticks); box.appendChild(ax);
    armsBox.appendChild(box);
    arms.push({d:d, lanes:lanes, ticks:ticks, t0:0});
  });

  // ---- render ------------------------------------------------------------
  function render(){
    arms.forEach(function(a){
      var W = Math.round(a.lanes[0].parentNode.getBoundingClientRect().width) || 800;
      var sc = W / view.span, v0 = a.t0, v1 = a.t0 + view.span;
      a.lanes.forEach(function(svg, si){
        svg.setAttribute('viewBox', '0 0 ' + W + ' ' + LANE);
        var f = document.createDocumentFragment();
        var bg = document.createElementNS(SVGNS,'rect');
        bg.setAttribute('class','tg-bg'); bg.setAttribute('x',0); bg.setAttribute('y',LANE-1);
        bg.setAttribute('width',W); bg.setAttribute('height',1); f.appendChild(bg);
        a.d.steps.forEach(function(st){
          [st.t0, st.t1].forEach(function(x){
            if (x <= v0 || x >= v1) return;
            var ln = document.createElementNS(SVGNS,'line');
            ln.setAttribute('class','tg-stepline');
            ln.setAttribute('x1',(x-v0)*sc); ln.setAttribute('x2',(x-v0)*sc);
            ln.setAttribute('y1',0); ln.setAttribute('y2',LANE); f.appendChild(ln);
          });
        });
        a.d.tasks.forEach(function(t, ti){
          if (t.s !== si) return;
          if (t.t0 + t.d < v0 || t.t0 > v1) return;
          var c = a.d.name_class[t.n];
          var x = (t.t0 - v0) * sc, w = Math.max(MINW, t.d * sc);
          // host-func node is 0.007 us: draw it as a tick, but let the tick shrink
          // with the zoom so 80 of them cannot out-ink the waits they sit on.
          if (c === 0) w = Math.min(2.2, Math.max(MINW, sc * 40));
          var xa = Math.max(0, x), xb = Math.min(W, x + w);   // clip to the lane
          if (xb <= 0 || xa >= W) return;
          var r = document.createElementNS(SVGNS,'rect');
          r.setAttribute('class','b c' + c);
          r.setAttribute('x', xa.toFixed(2));
          r.setAttribute('width', Math.max(MINW, xb - xa).toFixed(2));
          r.setAttribute('y', c === 0 ? 0 : c === 3 ? TOP + BH/2 : TOP);
          r.setAttribute('height', c === 0 ? 3 : c === 3 ? BH/2 : BH);
          r.dataset.i = ti;
          f.appendChild(r);
        });
        svg.textContent = '';
        svg.appendChild(f);
      });
      // ticks
      a.ticks.textContent = '';
      var stepUs = niceStep(view.span);
      for (var t = Math.ceil(v0/stepUs)*stepUs; t <= v1; t += stepUs){
        var i = el('i', null, (t/1000).toFixed(stepUs < 1000 ? 2 : (stepUs < 10000 ? 1 : 0)) + ' ms');
        var p = (t - v0) / view.span * 100;
        i.style.left = p + '%';
        if (p < 1.5) i.style.transform = 'none';        // don't clip the first label
        a.ticks.appendChild(i);
      }
    });
  }
  function niceStep(span){
    var raw = span / 8, p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    return (n < 1.5 ? 1 : n < 3 ? 2 : n < 7 ? 5 : 10) * p;
  }

  // ---- views -------------------------------------------------------------
  function setView(which){
    if (which === 'all'){
      view = {span: Math.max.apply(null, arms.map(function(a){return a.d.span_us;}))};
      arms.forEach(function(a){ a.t0 = 0; });
    } else {
      var k = +which;
      view = {span: Math.max.apply(null, arms.map(function(a){return a.d.steps[k].t1 - a.d.steps[k].t0;}))};
      arms.forEach(function(a){ a.t0 = a.d.steps[k].t0; });
    }
    ROOT.querySelectorAll('.tg-ctl button').forEach(function(b){
      b.setAttribute('aria-pressed', String(b.dataset.view === String(which)));
    });
    render();
  }
  ROOT.querySelectorAll('.tg-ctl button').forEach(function(b){
    b.addEventListener('click', function(){ setView(b.dataset.view); });
  });

  // ---- hover -------------------------------------------------------------
  armsBox.addEventListener('mousemove', function(ev){
    var r = ev.target;
    if (!(r instanceof SVGRectElement) || r.dataset.i === undefined){ tip.hidden = true; return; }
    var a = arms[+r.ownerSVGElement.dataset.arm], t = a.d.tasks[+r.dataset.i];
    var nm = a.d.names[t.n], c = a.d.name_class[t.n];
    tip.innerHTML = '';
    tip.appendChild(el('b', null, nm));
    var d2 = el('div'); d2.appendChild(el('em', null, '时长 ')); d2.appendChild(document.createTextNode(fmt(t.d)));
    d2.appendChild(el('em', null, '　起点 ')); d2.appendChild(document.createTextNode(fmt(t.t0)));
    tip.appendChild(d2);
    if (t.k) tip.appendChild(el('div', null, '（' + t.k + ' 个同类相邻任务合并成一块，块宽 = 它们真实跨度）'));
    if (c === 0) tip.appendChild(el('div', null, '（host-func 节点本身 ~0.007 µs，这里画成竖标）'));
    tip.hidden = false;
    var w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.min(ev.clientX + 12, innerWidth - w - 8) + 'px';
    tip.style.top = (ev.clientY - h - 12 < 8 ? ev.clientY + 16 : ev.clientY - h - 12) + 'px';
  });
  armsBox.addEventListener('mouseleave', function(){ tip.hidden = true; });

  // ---- brush zoom --------------------------------------------------------
  var drag = null;
  armsBox.addEventListener('pointerdown', function(ev){
    var svg = ev.target.closest ? ev.target.closest('svg.tg-svg') : null;
    if (!svg) return;
    var box = svg.getBoundingClientRect();
    drag = {svg:svg, box:box, a:arms[+svg.dataset.arm], x0:ev.clientX - box.left, rect:null};
    svg.setPointerCapture(ev.pointerId);
    ev.preventDefault();
  });
  armsBox.addEventListener('pointermove', function(ev){
    if (!drag) return;
    var x = Math.max(0, Math.min(drag.box.width, ev.clientX - drag.box.left));
    if (!drag.rect){
      drag.rect = document.createElementNS(SVGNS,'rect');
      drag.rect.setAttribute('class','tg-brush');
      drag.rect.setAttribute('y', 0); drag.rect.setAttribute('height', LANE);
      drag.svg.appendChild(drag.rect);
    }
    drag.rect.setAttribute('x', Math.min(drag.x0, x));
    drag.rect.setAttribute('width', Math.abs(x - drag.x0));
  });
  function endDrag(ev){
    if (!drag) return;
    var x = Math.max(0, Math.min(drag.box.width, ev.clientX - drag.box.left));
    var lo = Math.min(drag.x0, x), hi = Math.max(drag.x0, x), W = drag.box.width;
    var a = drag.a;
    if (hi - lo > 6){
      var newT0 = a.t0 + lo / W * view.span, newSpan = (hi - lo) / W * view.span;
      var shift = newT0 - a.t0;
      arms.forEach(function(o){ o.t0 = o.t0 + shift; });
      view.span = Math.max(newSpan, 20);
      ROOT.querySelectorAll('.tg-ctl button').forEach(function(b){ b.setAttribute('aria-pressed','false'); });
    }
    drag = null;
    render();
  }
  armsBox.addEventListener('pointerup', endDrag);
  armsBox.addEventListener('pointercancel', function(){ drag = null; render(); });
  armsBox.addEventListener('dblclick', function(){ setView('all'); });

  // ---- checks table ------------------------------------------------------
  var tb = ROOT.querySelector('.tg-tbody');
  CHECKS.forEach(function(r){
    var tr = el('tr');
    tr.appendChild(el('td', null, r[0]));
    tr.appendChild(el('td', null, r[1]));
    tr.appendChild(el('td','ok', r[2]));
    tr.appendChild(el('td', null, r[3]));
    tb.appendChild(tr);
  });

  var raf;
  addEventListener('resize', function(){ cancelAnimationFrame(raf); raf = requestAnimationFrame(render); });
  setView('all');
})();
</script>
</section>
'''


TITLES = {
    "serial": "串行臂 — 侧流关（P1a/P1b 均关）",
    "side": "侧流臂 — 侧流开、P1a 关",
}


def find(doc, role, typ, field):
    for sid, v in doc["checks"]["per_stream"].items():
        if v["role"] == role and typ in v["types"]:
            return v["types"][typ][field]
    return None


def ours(doc, role, typ, field):
    roles = {sid: v["role"] for sid, v in doc["checks"]["per_stream"].items()}
    for sid, d in doc["checks"]["layer_columns"].items():
        if roles.get(sid) == role and typ in d:
            return d[typ][field]
    return None


def build_checks(ser, sid):
    """[label, what the docs already carry, what this slice says, delta]"""
    def pct(a, b):
        return "%+.1f%%" % (100 * (a - b) / b) if b else "-"
    rows = []
    v = find(ser, "main", "HOSTFUNC_CALLBACK", "n_per_step")
    rows.append(["串行 · 主流 HOSTFUNC_CALLBACK 次/步",
                 "40（+2 非本层，§4.20②）", "%.0f（本层 %.0f）" % (v, ours(ser, "main", "HOSTFUNC_CALLBACK", "n_per_step_ours")),
                 "一致"])
    v = find(ser, "main", "HOSTFUNC_CALLBACK", "pregap_med_us")
    rows.append(["串行 · HOSTFUNC_CALLBACK 前置空隙中位", "30.75 µs（§4.22 F1）", "%.2f µs" % v, pct(v, 30.75)])
    v = ours(ser, "main", "NOTIFY_WAIT", "dur_med_us_ours")
    rows.append(["串行 · NOTIFY_WAIT 每层中位", "353.8 µs（§4.20）", "%.1f µs" % v, pct(v, 353.8)])
    v = find(sid, "main", "NOTIFY_WAIT", "n_per_step")
    rows.append(["侧流 · 主流 NOTIFY_WAIT 次/步", "2（§4.26）", "%.0f" % v, "一致"])
    v = find(sid, "side", "NOTIFY_WAIT", "n_per_step")
    rows.append(["侧流 · 副流 NOTIFY_WAIT 次/步", "40（§4.26）", "%.0f" % v, "一致"])
    v = find(sid, "side", "NOTIFY_WAIT", "dur_med_us")
    rows.append(["侧流 · 副流 NOTIFY_WAIT 中位", "333.5 µs（§4.26）", "%.1f µs" % v, pct(v, 333.5)])
    v = find(sid, "side", "HOSTFUNC_CALLBACK", "pregap_med_us")
    rows.append(["侧流 · 副流 HOSTFUNC_CALLBACK 前置空隙", "35.75 µs（§4.26）", "%.2f µs" % v, pct(v, 35.75)])
    v = find(sid, "main", "DAVID_EVENT_WAIT", "n_per_step")
    rows.append(["侧流 · 主流 DAVID_EVENT_WAIT 次/步", "200（§4.26）", "%.0f" % v, "一致"])
    v = find(sid, "main", "DAVID_EVENT_WAIT", "dur_ms_per_step")
    rows.append(["侧流 · 主流 DAVID_EVENT_WAIT 合计", "14.18 ms/step（整发均值）", "%.2f ms/step" % v,
                 pct(v, 14.18) + "；整发中位 13.42，稳态段 13.07 ⇒ 均值被 2 个 130 ms 卡顿步抬高"])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("serial")
    ap.add_argument("side")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args()
    docs = []
    for path in (a.serial, a.side):
        d = json.load(open(path))
        d["title"] = TITLES.get(d["arm"], d["arm"])
        docs.append(d)
    data = ",".join(json.dumps(d, separators=(",", ":"), ensure_ascii=False) for d in docs)
    html = TEMPLATE.replace("/*__DATA__*/", data)
    html = html.replace("/*__CHECKS__*/",
                        json.dumps(build_checks(docs[0], docs[1]), ensure_ascii=False,
                                   separators=(",", ":")))
    with open(a.out, "w") as f:
        f.write(html)
    print("-> %s  %.1f KB" % (a.out, len(html.encode()) / 1024))


if __name__ == "__main__":
    raise SystemExit(main())
