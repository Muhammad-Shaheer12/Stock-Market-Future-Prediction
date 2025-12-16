async function loadHealth(){
  const r = await fetch('/health');
  return await r.json();
}
async function predict(symbol){
  const r = await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol})});
  return await r.json();
}
async function backtest(){
  const sym = document.getElementById('symbol').value;
  const r = await fetch(`/backtest?symbol=${encodeURIComponent(sym)}`);
  return await r.json();
}
async function loadPrices(symbol){
  const url = `/prices?symbol=${encodeURIComponent(symbol)}&limit=60`;
  const r = await fetch(url);
  if(!r.ok){ throw new Error(`Prices fetch failed: ${r.status}`); }
  return await r.json();
}

async function loadDetails(symbol){
  const r = await fetch('/details',{
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({symbol})
  });
  if(!r.ok){
    const txt = await r.text();
    throw new Error(`Details fetch failed: ${r.status} ${txt}`);
  }
  return await r.json();
}

async function loadPca(symbol){
  const r = await fetch(`/dimred/pca?symbol=${encodeURIComponent(symbol)}`);
  if(!r.ok){
    const txt = await r.text();
    throw new Error(`PCA fetch failed: ${r.status} ${txt}`);
  }
  return await r.json();
}

async function loadCluster(symbol, k=4){
  const r = await fetch(`/cluster/assets?symbol=${encodeURIComponent(symbol)}&k=${encodeURIComponent(k)}`);
  if(!r.ok){
    const txt = await r.text();
    throw new Error(`Cluster fetch failed: ${r.status} ${txt}`);
  }
  return await r.json();
}

async function loadAssociations(horizon=1){
  const r = await fetch(`/associations?horizon=${encodeURIComponent(horizon)}`);
  if(!r.ok){
    const txt = await r.text();
    throw new Error(`Rules fetch failed: ${r.status} ${txt}`);
  }
  return await r.json();
}
function renderPredictions(preds){
  const el = document.getElementById('predictions');
  el.innerHTML = '';
  const order = ['1','3','7','30'];
  order.forEach(h=>{
    const obj = preds[h];
    const v = (obj && typeof obj === 'object') ? obj.predicted_log_return : obj;
    const winner = (obj && typeof obj === 'object') ? obj.winner_model : null;
    const card = document.createElement('div');
    card.className = 'kpi';
    const pct = (Number(v) * 100).toFixed(2);
    const winnerTxt = winner ? `<div class="sub">winner: ${winner}</div>` : '';
    card.innerHTML = `<div>H${h}</div><div><strong>${pct}%</strong></div>${winnerTxt}`;
    el.appendChild(card);
  });
}
function renderBacktest(bt){
  const tbody = document.querySelector('#backtest tbody');
  tbody.innerHTML = '';
  const rows = [
    ['Cumulative Return (log)', bt.cum_return?.toFixed(4)],
    ['Sharpe', bt.sharpe?.toFixed(2)],
    ['Max Drawdown', bt.max_drawdown?.toFixed(2)]
  ];
  rows.forEach(([k,v])=>{
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${k}</td><td>${v}</td>`;
    tbody.appendChild(tr);
  });
}
function renderChart(labels, values){
  const ctx = document.getElementById('chart').getContext('2d');
  if(window._chart) window._chart.destroy();
  const min = Math.min(...values);
  const max = Math.max(...values);
  const pad = (max - min) * 0.05;
  window._chart = new Chart(ctx,{
    type:'line',
    data:{labels,datasets:[{label:'Adjusted Close',data:values,borderColor:'#60a5fa',backgroundColor:'rgba(96,165,250,0.2)',tension:0.2,fill:true,pointRadius:0}]},
    options:{
      responsive:true,
      plugins:{
        legend:{labels:{color:'#e2e8f0'}},
        tooltip:{
          callbacks:{
            label: (ctx)=>{
              const v = ctx.parsed.y;
              return `$${v.toFixed(2)}`;
            }
          }
        }
      },
      scales:{
        x:{ticks:{color:'#93a3b8'}},
        y:{
          min: isFinite(min) ? min - pad : undefined,
          max: isFinite(max) ? max + pad : undefined,
          ticks:{
            color:'#93a3b8',
            callback: (v)=> `$${Number(v).toFixed(2)}`
          }
        }
      }
    }
  });
}

function setActiveView(view){
  const pred = document.getElementById('viewPred');
  const det = document.getElementById('viewDetails');
  if(!pred || !det) return;
  if(view === 'details'){
    pred.style.display = 'none';
    det.style.display = '';
  }else{
    det.style.display = 'none';
    pred.style.display = '';
  }
}

function fmtPct(x){
  if(x === null || x === undefined || !isFinite(Number(x))) return '—';
  return `${(Number(x) * 100).toFixed(2)}%`;
}

function renderHeatmap(canvasId, symbols, matrix){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const n = symbols.length;
  if(window._corrChart) window._corrChart.destroy();
  if(!n || !matrix || !matrix.length){
    window._corrChart = new Chart(ctx,{type:'bar',data:{labels:[],datasets:[]}});
    return;
  }
  const points = [];
  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      points.push({x:j,y:i,v:matrix[i][j]});
    }
  }
  const min = -1, max = 1;
  const colorFor = (v)=>{
    const t = (v - min) / (max - min);
    const r = Math.round(30 + 180*(1-t));
    const g = Math.round(60 + 120*(t));
    const b = 120;
    return `rgb(${r},${g},${b})`;
  };
  window._corrChart = new Chart(ctx,{
    type:'scatter',
    data:{datasets:[{
      label:'corr',
      data:points,
      pointRadius: (ctx)=>{
        const area = ctx.chart.chartArea;
        if(!area) return 6;
        const w = (area.right - area.left) / n;
        const h = (area.bottom - area.top) / n;
        return Math.max(5, Math.min(w,h) * 0.45);
      },
      pointHoverRadius: 8,
      backgroundColor: (c)=> colorFor(c.raw.v)
    }]},
    options:{
      responsive:true,
      plugins:{
        legend:{display:false},
        tooltip:{
          callbacks:{
            label: (c)=>{
              const i = c.raw.y, j = c.raw.x;
              return `${symbols[i]} vs ${symbols[j]}: ${Number(c.raw.v).toFixed(2)}`;
            }
          }
        }
      },
      scales:{
        x:{
          type:'linear',
          min:-0.5,max:n-0.5,
          ticks:{
            stepSize:1,
            color:'#93a3b8',
            callback:(v)=> symbols[Number(v)] || ''
          },
          grid:{color:'rgba(148,163,184,0.15)'}
        },
        y:{
          type:'linear',
          min:-0.5,max:n-0.5,
          ticks:{
            stepSize:1,
            color:'#93a3b8',
            callback:(v)=> symbols[Number(v)] || ''
          },
          grid:{color:'rgba(148,163,184,0.15)'}
        }
      }
    }
  });
}

function renderLineChart(canvasId, label, labels, values, color){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const key = `_chart_${canvasId}`;
  if(window[key]) window[key].destroy();
  window[key] = new Chart(ctx,{
    type:'line',
    data:{labels,datasets:[{label,data:values,borderColor:color,backgroundColor:'rgba(96,165,250,0.18)',tension:0.2,fill:true,pointRadius:0}]},
    options:{
      responsive:true,
      plugins:{legend:{labels:{color:'#e2e8f0'}}},
      scales:{x:{ticks:{color:'#93a3b8'}},y:{ticks:{color:'#93a3b8'}}}
    }
  });
}

function renderAtrDd(canvasId, atr, dd){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  if(window._atrDd) window._atrDd.destroy();
  const labels = (dd.dates||[]);
  const ddVals = (dd.values||[]);
  // Align ATR to drawdown length by using its own dates on a second axis; keep simple.
  window._atrDd = new Chart(ctx,{
    type:'line',
    data:{
      labels,
      datasets:[
        {label:'Drawdown',data:ddVals,borderColor:'#f97316',backgroundColor:'rgba(249,115,22,0.12)',tension:0.2,fill:true,pointRadius:0,yAxisID:'y'},
        {label:'ATR (14)',data:(atr.values||[]),borderColor:'#22c55e',backgroundColor:'rgba(34,197,94,0.10)',tension:0.2,fill:false,pointRadius:0,yAxisID:'y2'}
      ]
    },
    options:{
      responsive:true,
      plugins:{legend:{labels:{color:'#e2e8f0'}}},
      scales:{
        x:{ticks:{color:'#93a3b8'}},
        y:{ticks:{color:'#93a3b8'},title:{display:true,text:'Drawdown',color:'#93a3b8'}},
        y2:{position:'right',grid:{drawOnChartArea:false},ticks:{color:'#93a3b8'},title:{display:true,text:'ATR',color:'#93a3b8'}}
      }
    }
  });
}

function renderIntervals(canvasId, intervals){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  if(window._intervals) window._intervals.destroy();
  const hs = Object.keys(intervals||{}).sort((a,b)=>Number(a)-Number(b));
  const pred = hs.map(h=> intervals[h].pred);
  const lo = hs.map(h=> intervals[h].lo);
  const hi = hs.map(h=> intervals[h].hi);
  window._intervals = new Chart(ctx,{
    type:'line',
    data:{
      labels: hs.map(h=>`H${h}`),
      datasets:[
        {label:'Pred',data:pred,borderColor:'#60a5fa',tension:0.2,pointRadius:3},
        {label:'Low',data:lo,borderColor:'rgba(148,163,184,0.7)',tension:0.2,pointRadius:0},
        {label:'High',data:hi,borderColor:'rgba(148,163,184,0.7)',tension:0.2,pointRadius:0}
      ]
    },
    options:{
      responsive:true,
      plugins:{
        legend:{labels:{color:'#e2e8f0'}},
        tooltip:{callbacks:{label:(c)=> `${c.dataset.label}: ${(Number(c.parsed.y)*100).toFixed(2)}%`}}
      },
      scales:{x:{ticks:{color:'#93a3b8'}},y:{ticks:{color:'#93a3b8',callback:(v)=>`${(Number(v)*100).toFixed(1)}%`}}}
    }
  });
}

function renderPca(canvasId, pca){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  if(window._pcaChart) window._pcaChart.destroy();

  const labels = pca?.labels || [];
  const pts = (pca?.points||[]).map((xy,i)=>({x:Number(xy[0]),y:Number(xy[1]),label:labels[i]}));
  if(!pts.length){
    window._pcaChart = new Chart(ctx,{type:'scatter',data:{datasets:[]}});
    return;
  }
  window._pcaChart = new Chart(ctx,{
    type:'scatter',
    data:{datasets:[{label:'PCA',data:pts,backgroundColor:'rgba(96,165,250,0.45)',borderColor:'#60a5fa',pointRadius:3}]},
    options:{
      responsive:true,
      plugins:{
        legend:{labels:{color:'#e2e8f0'}},
        tooltip:{callbacks:{label:(c)=>{
          const d = c.raw;
          const ds = d.label ? ` (${d.label})` : '';
          return `(${Number(d.x).toFixed(2)}, ${Number(d.y).toFixed(2)})${ds}`;
        }}}
      },
      scales:{
        x:{ticks:{color:'#93a3b8'},grid:{color:'rgba(148,163,184,0.15)'}},
        y:{ticks:{color:'#93a3b8'},grid:{color:'rgba(148,163,184,0.15)'}}
      }
    }
  });
}

function renderCluster(canvasId, cluster){
  const canvas = document.getElementById(canvasId);
  if(!canvas) return;
  const ctx = canvas.getContext('2d');
  const labels = cluster?.labels || [];
  const k = Number(cluster?.k || 0);
  if(window._clusterChart) window._clusterChart.destroy();

  if(!labels.length || !k){
    window._clusterChart = new Chart(ctx,{type:'bar',data:{labels:[],datasets:[]}});
    return;
  }

  const counts = Array.from({length:k}, ()=>0);
  labels.forEach((x)=>{
    const i = Number(x);
    if(Number.isInteger(i) && i>=0 && i<k) counts[i] += 1;
  });
  const xs = counts.map((_,i)=>`C${i}`);
  window._clusterChart = new Chart(ctx,{
    type:'bar',
    data:{labels:xs,datasets:[{label:'Days per cluster',data:counts,backgroundColor:'rgba(34,197,94,0.45)',borderColor:'#22c55e',borderWidth:1}]},
    options:{
      responsive:true,
      plugins:{legend:{labels:{color:'#e2e8f0'}}},
      scales:{x:{ticks:{color:'#93a3b8'}},y:{ticks:{color:'#93a3b8'},grid:{color:'rgba(148,163,184,0.15)'}}}
    }
  });

  const el = document.getElementById('clusterSummary');
  if(el){
    const n = labels.length;
    const pct = counts.map(c=> n? `${Math.round((c/n)*100)}%`:'0%');
    el.textContent = `k=${k} over ${n} days • distribution: ${pct.join(', ')}`;
  }
}

function renderRules(rules){
  const tbody = document.querySelector('#rulesTable tbody');
  if(!tbody) return;
  tbody.innerHTML = '';
  const rows = (rules?.rules || []).slice(0, 20);
  rows.forEach((r)=>{
    const tr = document.createElement('tr');
    const ants = (r.antecedents||[]).join(', ');
    const cons = (r.consequents||[]).join(', ');
    const support = (Number(r.support)||0).toFixed(3);
    const conf = (Number(r.confidence)||0).toFixed(3);
    const lift = (Number(r.lift)||0).toFixed(3);
    tr.innerHTML = `<td>${ants}</td><td>${cons}</td><td>${support}</td><td>${conf}</td><td>${lift}</td>`;
    tbody.appendChild(tr);
  });
  if(!rows.length){
    const tr = document.createElement('tr');
    tr.innerHTML = `<td colspan="5" class="muted">No rules returned.</td>`;
    tbody.appendChild(tr);
  }
}

function renderDetails(d){
  document.getElementById('kpiBeta').textContent = (Number(d.beta_vs_spy)||0).toFixed(2);
  document.getElementById('kpiMdd').textContent = fmtPct(d.drawdown?.max_drawdown);
  const level = d.prediction_intervals ? (Object.values(d.prediction_intervals)[0]?.level) : null;
  document.getElementById('kpiInt').textContent = level ? `${Math.round(Number(level)*100)}%` : '—';

  const corr = d.correlation || {};
  renderHeatmap('corrHeatmap', corr.symbols||[], corr.matrix||[]);

  const vol = d.volatility || {};
  renderLineChart('volChart','Volatility',vol.dates||[],vol.annualized||[],'#60a5fa');

  renderAtrDd('atrDdChart', d.atr||{}, d.drawdown||{});
  renderIntervals('intervalsChart', d.prediction_intervals||{});
}
async function main(){
  const input = document.getElementById('symbol');
  const btn = document.getElementById('predict');
  const navPred = document.getElementById('navPred');
  const navDetails = document.getElementById('navDetails');
  const more = document.getElementById('more');
  const refreshDetails = document.getElementById('refreshDetails');
  const loadRulesBtn = document.getElementById('loadRules');
  const detailsStatus = document.getElementById('detailsStatus');
  let busy = false;
  let detailsBusy = false;
  let rulesBusy = false;

  const goDetails = async ()=>{
    if(detailsBusy) return;
    detailsBusy = true;
    const sym = input.value.trim()||'AAPL';
    setActiveView('details');
    if(detailsStatus) detailsStatus.textContent = `Loading details for ${sym}...`;
    if(navDetails) navDetails.disabled = true;
    if(more) more.disabled = true;
    if(refreshDetails) refreshDetails.disabled = true;
    try{
      const d = await loadDetails(sym);
      renderDetails(d);
      if(detailsStatus) detailsStatus.textContent = `Loaded details for ${sym}. Loading PCA and clustering...`;
      try{
        const [pca, cluster] = await Promise.all([
          loadPca(sym),
          loadCluster(sym, 4)
        ]);
        renderPca('pcaChart', pca);
        renderCluster('clusterChart', cluster);
        if(detailsStatus) detailsStatus.textContent = `Loaded details for ${sym}.`;
      }catch(e){
        console.error(e);
        if(detailsStatus) detailsStatus.textContent = `Loaded details for ${sym}, but PCA/cluster failed: ${e?.message || e}`;
      }
    }catch(e){
      console.error(e);
      if(detailsStatus) detailsStatus.textContent = `Failed to load details: ${e?.message || e}`;
    }
    finally{
      if(navDetails) navDetails.disabled = false;
      if(more) more.disabled = false;
      if(refreshDetails) refreshDetails.disabled = false;
      detailsBusy = false;
    }
  };

  if(navPred) navPred.addEventListener('click', ()=> setActiveView('pred'));
  if(navDetails) navDetails.addEventListener('click', ()=> goDetails());
  if(more) more.addEventListener('click', ()=> goDetails());
  if(refreshDetails) refreshDetails.addEventListener('click', ()=> goDetails());

  if(loadRulesBtn) loadRulesBtn.addEventListener('click', async ()=>{
    if(rulesBusy) return;
    rulesBusy = true;
    loadRulesBtn.disabled = true;
    const prev = loadRulesBtn.textContent;
    loadRulesBtn.textContent = 'Loading...';
    if(detailsStatus) detailsStatus.textContent = 'Loading association rules (may take longer on first run)...';
    try{
      const r = await loadAssociations(1);
      renderRules(r);
      if(detailsStatus) detailsStatus.textContent = 'Loaded association rules.';
    }catch(e){
      console.error(e);
      if(detailsStatus) detailsStatus.textContent = `Failed to load rules: ${e?.message || e}`;
    }finally{
      loadRulesBtn.disabled = false;
      loadRulesBtn.textContent = prev;
      rulesBusy = false;
    }
  });

  btn.addEventListener('click', async ()=>{
    if(busy || btn.disabled) return;
    busy = true;
    btn.disabled = true; btn.textContent = 'Loading...';
    try{
      const sym = input.value.trim()||'AAPL';
      const p = await predict(sym);
      renderPredictions(p.predicted_log_returns||{});
      const bt = await backtest();
      renderBacktest(bt);
      const pr = await loadPrices(sym);
      renderChart(pr.labels||[], pr.values||[]);
      setActiveView('pred');
    }catch(e){ console.error(e); }
    finally{ btn.disabled = false; btn.textContent = 'Get Predictions'; busy = false; }
  });
}
window.addEventListener('DOMContentLoaded', main);
