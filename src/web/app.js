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
async function main(){
  const input = document.getElementById('symbol');
  const btn = document.getElementById('predict');
  let busy = false;
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
    }catch(e){ console.error(e); }
    finally{ btn.disabled = false; btn.textContent = 'Get Predictions'; busy = false; }
  });
}
window.addEventListener('DOMContentLoaded', main);
