"use strict";
const $ = id => document.getElementById(id);
const COLORS = ["#078589","#b57d17","#8c60bd","#c65259","#52874c","#567fbd","#bd7e50","#7972b6","#237d63","#a44e87","#697329","#526075"];
const state = {catalog:[], selected:new Set(), run:null, step:0, cell:null, timer:null, tab:"replay", token:"", defaults:null, inspectSerial:0, loadSerial:0, compareSerial:0, aggregateSerial:0, series:new Map(), aggregates:[], jobs:[], jobsSignature:""};
const esc = s => String(s ?? "").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const num = (v,n=2) => v == null ? "—" : Number(v).toLocaleString(undefined,{maximumFractionDigits:n,minimumFractionDigits:n});
const short = r => r.method+" · "+r.mode+" · c="+r.multiplier+" · f="+(r.noise_factor??"oracle")+" · m"+r.map_seed+"/o"+r.observation_seed+" · H="+r.horizon;
const phase = v => (v || "—").replaceAll("_"," ");
function showError(e){$("notice").hidden=false;$("notice").textContent=e.message||String(e);}
function clearError(){$("notice").hidden=true;}
async function api(path,body){
  const options = body===undefined?{}:{method:"POST",headers:{"Content-Type":"application/json","X-Lab-Token":state.token},body:JSON.stringify(body)};
  const response=await fetch(path,options);const data=await response.json();
  if(!response.ok)throw Error(data.error||"Request failed");return data;
}
function handle(fn){return async(...args)=>{try{await fn(...args);}catch(e){showError(e);}};}
function tab(name){
  state.tab=name;
  document.querySelectorAll(".tab").forEach(e=>e.hidden=e.id!=="tab-"+name);
  document.querySelectorAll(".nav").forEach(e=>e.classList.toggle("active",e.dataset.tab===name));
  $("page-title").textContent={replay:"Understand every decision.",compare:"Evidence, side by side.",experiment:"Turn questions into experiments."}[name];
  if(name!=="replay")stop();
  if(name==="compare"){drawSelected().catch(showError);if(!state.aggregates.length)analyse().catch(showError);}
}
async function refresh(){
  state.catalog=await api("/api/catalog?refresh=1");state.series.clear();renderLibrary();updateBatchOptions();invalidateAnalysis();
}
function renderLibrary(){
  const q=$("search").value.toLowerCase(), group=$("group").value;
  const records=state.catalog.filter(r=>(group==="all"||r.group===group)&&JSON.stringify([r.label,r.path,r.map_seed,r.observation_seed,r.method,r.mode]).toLowerCase().includes(q));
  $("library-count").textContent=records.length;
  $("library").innerHTML=records.map(r=>'<div class="run-item '+(state.run?.record.id===r.id?"selected":"")+'"><input type="checkbox" data-select="'+r.id+'" aria-label="Compare '+esc(short(r))+'" '+(state.selected.has(r.id)?"checked":"")+'><button class="run-open" data-open="'+r.id+'">'+esc(r.label)+'<small>'+r.rows+' × '+r.cols+' · map '+r.map_seed+' · obs '+r.observation_seed+' · '+num(r.noise_factor,1)+'× noise</small><small>'+esc(r.path.replace("results/",""))+'</small></button></div>').join("")||'<p class="subtle">No matching runs yet.</p>';
  $("selected-count").textContent=state.selected.size;
}
async function loadRun(id){
  stop();clearError();const serial=++state.loadSerial;
  $("run-title").textContent="Loading recorded trajectory…";
  const run=await api("/api/run/"+id);if(serial!==state.loadSerial)return;
  state.run=run;state.step=0;state.cell=null;state.series.set(id,{record:run.record,frames:run.frames.map(f=>Object.fromEntries(["step","discounted_return","cumulative_reward","certified_cells","false_certified_actions","unsafe_actions","phase"].map(k=>[k,f[k]])))});
  state.fieldRanges=new Map();
  const r=run.record,c=run.config;
  $("run-title").textContent=r.label;
  $("run-meta").textContent=r.rows+" × "+r.cols+" · "+r.map_source+" map "+r.map_seed+" · observation "+r.observation_seed+" · "+r.method+" · "+r.mode+" · c="+r.multiplier+" · assumed noise "+num(c.assumed_noise_factor,1)+"× actual";
  $("run-content").hidden=false;$("use-config").disabled=false;
  $("step").max=run.frames.length-1;$("step").value=0;
  const unsafeIndex=run.frames.findIndex(f=>f.unsafe);
  state.firstUnsafe=unsafeIndex>0?unsafeIndex:null;
  $("jump-before").disabled=$("jump-unsafe").disabled=state.firstUnsafe===null;
  $("jump-switch").disabled=run.summary.transition_step===null;
  $("cell-x").max=c.rows-1;$("cell-y").max=c.cols-1;
  $("config-view").textContent=JSON.stringify({config:run.config,provenance:run.map.provenance,recorded_path:r.path},null,2);
  $("stop-badge").textContent=phase(run.summary.stop_reason);$("stop-badge").className="badge "+(r.failed?"red":"teal");
  $("downloads").innerHTML=[["config.json","Config JSON"],["summary.json","Summary JSON"],["progress.csv","Progress CSV"],["trajectory.json.gz","Raw trajectory"],["observations.json.gz","Observations"],["manifest.json","Source manifest"]].map(([p,t])=>'<a href="/artifact/'+r.id+'/'+p+'" download>'+t+' ↓</a>').join("");
  renderLibrary();tab("replay");renderStep();
}
function setStep(value){
  if(!state.run)return;
  state.step=Math.max(0,Math.min(state.run.frames.length-1,Math.round(value)));
  $("step").value=state.step;renderStep();
}
function stop(){if(state.timer)clearInterval(state.timer);state.timer=null;$("play").textContent="▶ Play";}
function play(){
  if(state.timer){stop();return;}if(!state.run)return;
  if(state.step===state.run.frames.length-1)setStep(0);
  $("play").textContent="Ⅱ Pause";
  state.timer=setInterval(()=>{if(state.step>=state.run.frames.length-1)stop();else setStep(state.step+1);},Number($("speed").value));
}
function metric(label,value,hint,danger=false){return '<div class="metric '+(danger?"danger":"")+'"><small>'+label+'</small><strong>'+value+'</strong><span class="hint">'+hint+'</span></div>';}
function renderStep(){
  const p=state.run,f=p.frames[state.step],c=p.config;
  $("step-label").textContent=f.step+" / "+(p.frames.length-1);$("phase").textContent=phase(f.phase);
  $("replay-metrics").innerHTML=metric("Discounted return",num(f.discounted_return),"γ = "+c.discount+" · true collected reward")+metric("Unsafe actions",num(f.unsafe_actions,0),"Cumulative physical constraint violations",f.unsafe_actions>0)+metric("Certified cells",num(f.certified_cells,0),p.map.rows*p.map.cols+" total cells")+metric("False certificates",num(f.false_certified_actions,0),"At this step · evaluator check",f.false_certified_actions>0);
  drawMap("belief-map",true);drawMap("field-map",false);
  $("belief-caption").textContent="Rover ("+f.position.join(", ")+") · next action: "+(f.next_action||"none")+" · "+f.certified_action_count+" certified directed actions";
  $("decision-text").textContent="Step "+f.step+": "+phase(f.executed_phase||"initial observation")+". Next planned action: "+(f.next_action||"none")+". Exploration switch: "+(p.summary.transition_step??"not reached")+" ("+phase(p.summary.transition_reason||"none")+"). Final stop: "+phase(p.summary.stop_reason)+".";
  const prior=state.step>0?p.frames[state.step-1]:null;
  $("step-evidence").textContent=(prior?"Arrived via "+f.action+" from ("+prior.position.join(", ")+"). Pre-action LCB: "+num(prior.next_action_safety_lower,5)+"; actual g: "+num(f.true_transition_safety,5)+"; h: "+c.safety_threshold+". "+(f.unsafe?"This move violated the constraint. ":"This move did not violate the constraint. "):"Initial noisy observations train the altitude and reward GPs. ")+(f.es2_outside_sources?"ES2: "+f.es2_outside_sources.length+" executable source states have an auxiliary-policy successor outside the executable region. ":"")+(f.transition_step===f.step?"Exploration stops here: "+phase(f.transition_reason)+". ":"")+"The phase badge describes the next decision; collected reward and violations describe actions already executed.";
  const cell=state.cell||f.position;$("cell-x").value=cell[0];$("cell-y").value=cell[1];
  const key=$("replay-metric").value;
  lineChart($("replay-chart"),[{label:short(p.record),frames:p.frames}],key,{step:state.step,transition:p.summary.transition_step,onClick:setStep});
  scheduleInspect();
}
function tooltip(event,text){const e=$("tooltip");e.textContent=text;e.hidden=false;e.style.left=Math.max(8,Math.min(innerWidth-e.offsetWidth-8,event.clientX+14))+"px";e.style.top=Math.max(8,Math.min(innerHeight-e.offsetHeight-8,event.clientY+14))+"px";}
function hideTooltip(){$("tooltip").hidden=true;}
function mapGeometry(canvas){
  const p=state.run.map,pad=38,cell=Math.min((canvas.width-2*pad)/p.rows,(canvas.height-2*pad)/p.cols);
  return{cell,x:(canvas.width-cell*p.rows)/2,y:(canvas.height-cell*p.cols)/2,rows:p.rows,cols:p.cols};
}
function mapCell(canvas,event){
  const rect=canvas.getBoundingClientRect(),g=mapGeometry(canvas),x=Math.floor(((event.clientX-rect.left)*canvas.width/rect.width-g.x)/g.cell),y=g.cols-1-Math.floor(((event.clientY-rect.top)*canvas.height/rect.height-g.y)/g.cell);
  return x>=0&&x<g.rows&&y>=0&&y<g.cols?[x,y]:null;
}
function fieldData(){
  const key=$("field").value,p=state.run;
  return key==="true_reward"?p.map.reward:key==="true_altitude"?p.map.altitude:p.frames[state.step][key];
}
function fieldColor(t){t=Math.max(0,Math.min(1,t));const a=[229,241,232],b=[14,107,119];return "rgb("+a.map((v,i)=>Math.round(v+(b[i]-v)*t)).join(",")+")";}
function fieldRange(field){
  const key=$("field").value,fixed=$("field-scale").value==="run";
  if(fixed&&state.fieldRanges.has(key))return state.fieldRanges.get(key);
  let lo=Infinity,hi=-Infinity;
  const matrices=fixed?(key.startsWith("true_")?[field]:state.run.frames.map(f=>f[key])):[field];
  for(const matrix of matrices)if(matrix)for(const row of matrix)for(const v of row)if(Number.isFinite(v)){lo=Math.min(lo,v);hi=Math.max(hi,v);}
  const range=lo===Infinity?[0,1]:[lo,hi];if(fixed)state.fieldRanges.set(key,range);return range;
}
function drawMap(id,belief){
  const canvas=$(id),ctx=canvas.getContext("2d"),p=state.run,f=p.frames[state.step],g=mapGeometry(canvas),field=fieldData(),[lo,hi]=fieldRange(field);
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle="#fff";ctx.fillRect(0,0,canvas.width,canvas.height);
  const center=(x,y)=>[g.x+(x+.5)*g.cell,g.y+(g.cols-y-.5)*g.cell];
  for(let x=0;x<g.rows;x++)for(let y=0;y<g.cols;y++){
    const left=g.x+x*g.cell,top=g.y+(g.cols-1-y)*g.cell;
    ctx.fillStyle=belief?(f.pessimistic[x][y]?"#4b9fa9":f.optimistic[x][y]?"#f1dbab":"#e4e9eb"):(field&&field[x]?.[y]!=null?fieldColor((field[x][y]-lo)/(hi-lo||1)):"#edf1f2");
    ctx.fillRect(left,top,g.cell,g.cell);ctx.strokeStyle="#ffffff70";ctx.lineWidth=.8;ctx.strokeRect(left,top,g.cell,g.cell);
    if(belief&&$("arrows").checked){const mask=f.certified_actions[x*g.cols+y];[[0,0],[0,1],[1,0],[0,-1],[-1,0]].forEach(([dx,dy],a)=>{if(!a||!mask[a])return;const [cx,cy]=center(x,y);ctx.strokeStyle="#ffffffa0";ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(cx+dx*g.cell*.12,cy-dy*g.cell*.12);ctx.lineTo(cx+dx*g.cell*.37,cy-dy*g.cell*.37);ctx.stroke();});}
  }
  ctx.lineWidth=Math.max(2,g.cell*.1);ctx.lineCap="round";
  for(const unsafeOnly of [false,true])for(let i=1;i<=state.step;i++){if(Boolean(p.frames[i].unsafe)!==unsafeOnly)continue;const a=p.frames[i-1].position,b=p.frames[i].position;ctx.strokeStyle=unsafeOnly?"#dd4545":"#183a4bb0";ctx.beginPath();ctx.moveTo(...center(...a));ctx.lineTo(...center(...b));ctx.stroke();}
  const [sx,sy]=center(...p.map.start);ctx.strokeStyle="#fff";ctx.lineWidth=2;ctx.strokeRect(sx-g.cell*.25,sy-g.cell*.25,g.cell*.5,g.cell*.5);
  if(f.target){const [tx,ty]=center(...f.target);ctx.strokeStyle="#9f62b6";ctx.lineWidth=2;ctx.strokeRect(tx-g.cell*.37,ty-g.cell*.37,g.cell*.74,g.cell*.74);}
  if(state.cell){const [tx,ty]=center(...state.cell);ctx.strokeStyle="#ed8a37";ctx.lineWidth=3;ctx.strokeRect(tx-g.cell*.48,ty-g.cell*.48,g.cell*.96,g.cell*.96);}
  const [rx,ry]=center(...f.position);ctx.beginPath();ctx.arc(rx,ry,Math.max(4,g.cell*.3),0,Math.PI*2);ctx.fillStyle=f.unsafe?"#d44640":"#163c49";ctx.fill();ctx.strokeStyle="#fff";ctx.lineWidth=2.5;ctx.stroke();
  ctx.fillStyle="#7a8c91";ctx.font="11px -apple-system,sans-serif";ctx.textAlign="center";
  const tick=Math.max(1,Math.ceil(Math.max(g.rows,g.cols)/10));
  for(let x=0;x<g.rows;x+=tick)ctx.fillText(x,center(x,0)[0],g.y+g.cols*g.cell+20);
  ctx.textAlign="right";for(let y=0;y<g.cols;y+=tick)ctx.fillText(y,g.x-10,center(0,y)[1]+4);
  if(!belief){$("field-caption").textContent=field?$("field").selectedOptions[0].text+" · range "+num(lo,3)+" to "+num(hi,3)+" · "+($("field-scale").value==="run"?"fixed for this run":"rescaled at each frame"):"No GP field is recorded for this method.";$("field-legend").textContent=num(lo,3)+"   →   "+num(hi,3);}
}
let inspectTimer;
function scheduleInspect(){clearTimeout(inspectTimer);const serial=++state.inspectSerial;$("inspector").textContent="Reconstructing action bounds from saved observations…";inspectTimer=setTimeout(()=>inspectCell().catch(e=>{if(serial===state.inspectSerial)$("inspector").textContent=e.message;}),180);}
async function inspectCell(){
  const serial=state.inspectSerial,p=state.run,s=state.step,cell=state.cell||p.frames[s].position;
  const data=await api("/api/inspect?run="+p.record.id+"&step="+s+"&x="+cell[0]+"&y="+cell[1]);
  if(serial!==state.inspectSerial)return;
  $("inspector-title").textContent="Action safety at ("+cell.join(", ")+") · step "+s+" · h="+data.threshold;
  $("inspector").innerHTML='<div class="table-wrap"><table><thead><tr><th>Action → cell</th><th>Mean g</th><th>Joint σ</th><th>LCB</th><th>True g</th><th>True safe?</th><th>Executable?</th><th>Support</th></tr></thead><tbody>'+data.actions.map(a=>'<tr><td>'+a.action+' → ('+a.target.join(", ")+')</td><td>'+num(a.mean,4)+'</td><td>'+num(a.std,4)+'</td><td>'+num(a.lower,4)+'</td><td>'+num(a.true_g,4)+'</td><td class="'+(a.true_safe===false?"text-red":"")+'">'+(a.true_safe==null?"boundary":a.true_safe?"yes":"no")+'</td><td class="'+(a.certified&&!a.true_safe?"text-red":"")+'">'+(a.certified?"yes":"no")+'</td><td>'+esc(a.support)+'</td></tr>').join("")+'</tbody></table></div>';
}
function lineChart(container,series,key,options={}){
  if(!series.length){container.innerHTML='<p class="empty">Choose recorded runs to display a graph.</p>';return;}
  const w=1000,h=290,pad={l:65,r:22,t:28,b:48},plotw=w-pad.l-pad.r,ploth=h-pad.t-pad.b;
  const xs=series.flatMap(s=>s.frames.map(f=>f.step)),ys=series.flatMap(s=>s.frames.map(f=>Number(f[key]))).filter(Number.isFinite);
  const xmax=Math.max(1,...xs),ymin=Math.min(0,...ys),ymax=Math.max(ymin+1,...ys),x=v=>pad.l+v/xmax*plotw,y=v=>h-pad.b-(v-ymin)/(ymax-ymin)*ploth;
  let svg='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="'+esc(phase(key))+' by physical action step">';
  svg+='<text x="'+pad.l+'" y="15" fill="#536e76" font-size="11">'+esc(phase(key))+'</text><text x="'+(w-pad.r)+'" y="'+(h-1)+'" text-anchor="end" fill="#536e76" font-size="11">Physical actions executed</text>';
  for(let i=0;i<=4;i++){const v=ymin+(ymax-ymin)*i/4;svg+='<line x1="'+pad.l+'" x2="'+(w-pad.r)+'" y1="'+y(v)+'" y2="'+y(v)+'" stroke="#e6edef"/><text x="'+(pad.l-12)+'" y="'+(y(v)+4)+'" text-anchor="end" fill="#7c8e94" font-size="11">'+num(v,v<10?1:0)+'</text>';}
  for(let i=0;i<=5;i++){const v=xmax*i/5;svg+='<text x="'+x(v)+'" y="'+(h-13)+'" text-anchor="middle" fill="#7c8e94" font-size="11">'+num(v,0)+'</text>';}
  if(options.transition!=null)svg+='<line x1="'+x(options.transition)+'" x2="'+x(options.transition)+'" y1="'+pad.t+'" y2="'+(h-pad.b)+'" stroke="#c2a267" stroke-dasharray="5 5"/>';
  series.forEach((s,i)=>{svg+='<path d="'+s.frames.filter(f=>Number.isFinite(Number(f[key]))).map((f,j)=>(j?"L":"M")+x(f.step).toFixed(2)+","+y(f[key]).toFixed(2)).join(" ")+'" fill="none" stroke="'+COLORS[i%COLORS.length]+'" stroke-width="2.5" stroke-linejoin="round"/>';});
  if(options.step!=null)svg+='<line x1="'+x(options.step)+'" x2="'+x(options.step)+'" y1="'+pad.t+'" y2="'+(h-pad.b)+'" stroke="#335462" opacity=".6"/>';
  svg+='<rect class="chart-hit" x="'+pad.l+'" y="'+pad.t+'" width="'+plotw+'" height="'+ploth+'" fill="transparent" style="cursor:'+(options.onClick?"crosshair":"default")+'"/></svg>';
  container.innerHTML=svg+'<div class="legend">'+series.map((s,i)=>'<span><i style="background:'+COLORS[i%COLORS.length]+'"></i>'+esc(s.label)+'</span>').join("")+'</div>';
  const hit=container.querySelector(".chart-hit");
  const nearest=e=>{const rect=container.querySelector("svg").getBoundingClientRect();return Math.round(Math.max(0,Math.min(xmax,((e.clientX-rect.left)*w/rect.width-pad.l)/plotw*xmax)));};
  hit.onmousemove=e=>{const step=nearest(e);tooltip(e,"Step "+step+"\n"+series.map(s=>{const f=s.frames.reduce((a,b)=>Math.abs(a.step-step)<=Math.abs(b.step-step)?a:b);return s.label+": "+num(f[key],3)+(f.step!==step?" (ends at "+f.step+")":"");}).join("\n"));};
  hit.onmouseleave=hideTooltip;if(options.onClick)hit.onclick=e=>options.onClick(nearest(e));
}
async function drawSelected(){
  const serial=++state.compareSerial;
  const records=[...state.selected].map(id=>state.catalog.find(r=>r.id===id)).filter(Boolean);
  if($("aggregate-scope").value==="selected")invalidateAnalysis();
  $("selected-count").textContent=records.length;
  $("selected-chips").innerHTML=records.map((r,i)=>'<span class="chip" style="border-left:3px solid '+COLORS[i%COLORS.length]+'">'+esc(short(r))+'<button data-remove="'+r.id+'" aria-label="Remove '+esc(short(r))+'">×</button></span>').join("");
  const data=await Promise.all(records.map(async r=>{if(!state.series.has(r.id))state.series.set(r.id,await api("/api/series/"+r.id));const s=state.series.get(r.id);return{label:short(r),frames:s.frames};}));
  if(serial!==state.compareSerial)return;
  lineChart($("comparison-chart"),data,$("comparison-metric").value);
  $("pair-detail").innerHTML="";
  if(records.length>=2){
    const p=await api("/api/compare",{ids:records.map(r=>r.id)});
    if(serial!==state.compareSerial)return;
    $("pair-detail").innerHTML=p.warnings.map(w=>'<p class="callout caution">'+esc(w)+'</p>').join("")+(records.length===2?'<p class="callout">'+(p.paired?'Left: '+esc(p.paired.left.label)+'; right: '+esc(p.paired.right.label)+'. Right − left: return '+num(p.paired.right_minus_left_return,3)+'; unsafe actions '+num(p.paired.right_minus_left_unsafe_actions,0)+'. One matched pair is descriptive evidence.':'These are not a matched pair under the map, noise-stream, source-revision, and scoring checks.')+'</p><details><summary>Configuration differences ('+p.changes.length+')</summary><pre>'+esc(JSON.stringify(p.changes,null,2))+'</pre></details>':'<p class="subtle">Varied settings: '+esc(p.varying_fields.join(", ")||"method only")+'. Curves show recorded prefixes; no reward or actions are fabricated beyond early termination.</p>');
  }
}
function updateBatchOptions(){
  const select=$("aggregate-batch"),previous=select.value;
  select.innerHTML='<option value="all">All batches</option>'+state.jobs.map(j=>'<option value="'+esc(j.id)+'">'+esc(j.id)+' · '+esc(j.kind)+' · '+j.completed+'/'+j.total+'</option>').join("");
  if([...select.options].some(o=>o.value===previous))select.value=previous;
}
function batchEvidence(){
  const scope=$("aggregate-scope").value,batch=$("aggregate-batch").value;
  $("aggregate-batch").disabled=scope!=="interactive";
  const batches=scope==="interactive"?state.jobs.filter(j=>batch==="all"||j.id===batch):[];
  const planned=batches.reduce((n,j)=>n+j.total,0),finished=batches.reduce((n,j)=>n+j.completed,0),failures=batches.flatMap(j=>j.cases).filter(c=>c.error).length;
  $("batch-evidence").textContent=batches.length?"Batch totals before map/noise filters: "+planned+" planned · "+finished+" finished · "+failures+" setup failures · "+(planned-finished)+" without a completed case result. Trajectory means below exclude cases with no trajectory; review failures before making a full-batch claim.":"";
  return {batches:batches.map(j=>({id:j.id,status:j.status,planned:j.total,finished:j.completed,setup_failures:j.cases.filter(c=>c.error).length})),planned,finished,setup_failures:failures};
}
function invalidateAnalysis(){
  state.aggregateSerial++;state.aggregates=[];state.analysisSnapshot=null;
  $("export-comparison").disabled=$("export-analysis").disabled=true;
  $("aggregate-summary").textContent="Choose filters, then press Update analysis.";
  $("aggregate-table").innerHTML="";$("tradeoff-chart").innerHTML="";$("analysis-warnings").innerHTML="";
  batchEvidence();
}
async function analyse(){
  invalidateAnalysis();const serial=state.aggregateSerial;$("aggregate-summary").textContent="Updating analysis…";
  const scope=$("aggregate-scope").value,noise=$("aggregate-noise").value,m=$("aggregate-map").value,o=$("aggregate-observation").value;
  const batch=$("aggregate-batch").value;
  const records=state.catalog.filter(r=>(scope==="selected"?state.selected.has(r.id):r.group===scope)&&(scope!=="interactive"||batch==="all"||r.batch===batch)&&(noise==="all"||r.method==="oracle"||r.noise_factor===Number(noise))&&(m===""||r.map_seed===Number(m))&&(o===""||r.observation_seed===Number(o)));
  if(!records.length){$("aggregate-table").textContent="No runs match these filters.";$("aggregate-summary").textContent="";$("tradeoff-chart").innerHTML="";state.aggregates=[];$("export-comparison").disabled=true;return;}
  const d=await api("/api/compare",{ids:records.map(r=>r.id)});
  if(serial!==state.aggregateSerial)return;
  state.aggregates=d.aggregates;$("export-comparison").disabled=$("export-analysis").disabled=false;
  state.analysisSnapshot={created:new Date().toISOString(),filters:{scope,noise_factor:noise,map_seed:m,observation_seed:o,batch},batch_context:batchEvidence(),analysis:d,runs:records.map(r=>({id:r.id,path:r.path,config:r.config,method:r.method,map_hash:r.map_hash,model_hash:r.model_hash}))};
  $("analysis-warnings").innerHTML=d.warnings.map(w=>'<p class="callout caution">'+esc(w)+'</p>').join("");
  $("aggregate-summary").innerHTML='<p class="subtle">'+d.unique_runs+' unique recorded runs · '+d.duplicates_excluded+' duplicate copies excluded · '+d.aggregates.length+' conditions. Scope: '+esc(scope)+'; noise factor: '+esc(noise)+'; map: '+esc(m||"all")+'; observation: '+esc(o||"all")+'. Recorded early terminations remain included.</p>';
  $("aggregate-table").innerHTML='<table><thead><tr><th>Condition / inspect settings</th><th>Noise factor</th><th>Runs / maps</th><th>Return mean ± map SD</th><th>Mean unsafe actions</th><th>Runs with violation*</th><th>Early termination*</th><th>ES2 triggered*</th></tr></thead><tbody>'+d.aggregates.map(r=>'<tr><td><strong>'+r.condition_id+'</strong> · '+esc(r.condition)+'<br><small>'+esc(r.mode)+' · c='+r.multiplier+' · H='+r.horizon+'</small><details><summary>Exact settings &amp; cohort</summary><pre>'+esc(JSON.stringify({configuration:JSON.parse(r.configuration),model_hash:r.model_hash,cohort:r.cohort,mean_peak_false_certificates:r.false_certificates_mean,mean_exploration_actions:r.exploration_actions_mean},null,2))+'</pre></details></td><td>'+num(r.noise_factor,1)+'</td><td>'+r.runs+' / '+r.maps+'</td><td class="numeric">'+num(r.return_mean)+' ± '+num(r.return_sd)+'</td><td class="'+(r.unsafe_actions_mean>0?"text-red":"text-teal")+'">'+num(r.unsafe_actions_mean,3)+'</td><td>'+num(100*r.unsafe_fraction,1)+'%</td><td>'+num(100*r.failed_mean,1)+'%</td><td>'+num(100*r.es2_triggered_mean,1)+'%</td></tr>').join("")+'</tbody></table><p class="subtle">* Fractions are averaged within map, then equally across maps. Condition IDs connect the table to the scatter plot.</p>';
  tradeoff(d.aggregates);
}
function tradeoff(rows){
  const w=1000,h=280,pad={l:70,r:35,t:25,b:50},xmax=Math.max(.01,...rows.map(r=>r.unsafe_actions_mean))*1.15,ymax=Math.max(1,...rows.map(r=>r.return_mean))*1.15,x=v=>pad.l+v/xmax*(w-pad.l-pad.r),y=v=>h-pad.b-v/ymax*(h-pad.t-pad.b);
  let s='<svg viewBox="0 0 '+w+' '+h+'" role="img" aria-label="Mean return versus mean unsafe actions">';
  for(let i=0;i<=4;i++){const yy=ymax*i/4;s+='<line x1="'+pad.l+'" x2="'+(w-pad.r)+'" y1="'+y(yy)+'" y2="'+y(yy)+'" stroke="#e6edef"/><text x="'+(pad.l-10)+'" y="'+(y(yy)+4)+'" text-anchor="end" font-size="11" fill="#7c8e94">'+num(yy,0)+'</text>';const xx=xmax*i/4;s+='<text x="'+x(xx)+'" y="'+(h-pad.b+20)+'" text-anchor="middle" font-size="11" fill="#7c8e94">'+num(xx,2)+'</text>';}
  rows.forEach((r,i)=>{const nearby=rows.filter(a=>Math.abs(x(a.unsafe_actions_mean)-x(r.unsafe_actions_mean))<3&&Math.abs(y(a.return_mean)-y(r.return_mean))<3);s+='<circle cx="'+x(r.unsafe_actions_mean)+'" cy="'+y(r.return_mean)+'" r="7" fill="'+COLORS[i%COLORS.length]+'" stroke="white" stroke-width="2"><title>'+esc(nearby.map(a=>a.condition_id+": "+a.condition+" · factor "+a.noise_factor+"\nMean unsafe: "+num(a.unsafe_actions_mean,3)+"; mean return: "+num(a.return_mean)).join("\nCoincident point — "))+'</title></circle>';});
  s+='<text x="'+(w/2)+'" y="'+(h-5)+'" text-anchor="middle" font-size="11" fill="#7c8e94">Mean unsafe actions per run →</text><text x="15" y="18" font-size="11" fill="#7c8e94">Mean return ↑</text></svg>';
  $("tradeoff-chart").innerHTML=s+'<div class="legend">'+rows.map((r,i)=>'<span><i style="background:'+COLORS[i%COLORS.length]+'"></i>'+r.condition_id+' · '+esc(r.condition)+' · '+num(r.noise_factor,1)+'×</span>').join("")+'</div>';
}
function checked(name){return [...document.querySelectorAll('input[name="'+name+'"]:checked')].map(e=>e.value);}
function applyConfig(config){
  for(const[k,v]of Object.entries(config)){const e=document.querySelector('#experiment-form [name="'+k+'"]');if(e)e.value=v;}
  $("sweep-maps").value=config.map_seed;$("sweep-observations").value=config.observation_seed;
  $("kind").value="single";updateExperimentMode();
}
function invalidatePlan(){$("plan-count").textContent="";$("plan-detail").hidden=true;$("plan-view").textContent="";}
function updateExperimentMode(){
  const kind=$("kind").value;$("matched-options").hidden=kind!=="matched";$("sweep-options").hidden=kind!=="sweep";
  $("method").disabled=kind!=="single";
  for(const name of ["safety_confidence_multiplier","certificate_mode"])document.querySelector('[name="'+name+'"]').disabled=kind!=="single";
  for(const name of ["map_seed","observation_seed","assumed_noise_factor"])document.querySelector('[name="'+name+'"]').disabled=kind==="sweep";
  $("effective-controls").textContent=kind==="matched"?"Method, safety c, and certificate mode are set by the checked matched conditions.":kind==="sweep"?"Map/observation seeds, c, assumed-noise factors, modes, and methods are set by the sweep controls.":"All displayed single-run settings apply.";
  invalidatePlan();
}
function readConfig(){
  const config={...state.defaults};
  for(const k of Object.keys(config)){const e=document.querySelector('#experiment-form [name="'+k+'"]');if(e)config[k]=e.tagName==="SELECT"?e.value:Number(e.value);}
  return config;
}
function numbers(id){return $(id).value.split(",").map(s=>s.trim()).filter(Boolean).map(Number);}
function request(){
  const kind=$("kind").value,r={kind,config:readConfig(),method:$("method").value};
  if(kind==="matched")r.presets=checked("comparison-preset");
  if(kind==="sweep")Object.assign(r,{map_seeds:numbers("sweep-maps"),observation_seeds:numbers("sweep-observations"),multipliers:numbers("sweep-multipliers"),noise_factors:numbers("sweep-factors"),certificate_modes:checked("sweep-mode"),methods:checked("sweep-method")});
  return r;
}
function preset(name){
  const c={...state.defaults};
  if(name==="revalidated"){c.certificate_mode="current_lcb";c.safety_confidence_multiplier=4;}
  if(name==="smoke")Object.assign(c,{rows:6,cols:6,map_source:"generated",map_seed:3,observation_seed:2026,horizon:20,discount:.95});
  applyConfig(c);$("method").value="sno_es2";clearError();
}
async function preview(){
  if(!$("experiment-form").reportValidity())return;
  clearError();const p=await api("/api/plan",request());$("plan-count").textContent=p.count+" case"+(p.count===1?"":"s")+" planned";$("plan-detail").hidden=false;$("plan-view").textContent=JSON.stringify(p,null,2);return p;
}
async function pollJobs(){
  if(state.polling)return;state.polling=true;
  try{
  const jobs=await api("/api/jobs"),signature=JSON.stringify(jobs);
  if(signature===state.jobsSignature)return;
  state.jobsSignature=signature;state.jobs=jobs;
  await refresh();renderJobs();
  }finally{state.polling=false;}
}
function renderJobs(){
  $("jobs").innerHTML=state.jobs.map(j=>'<div class="job"><div class="job-title"><h4>'+esc(j.id)+' · '+esc(j.kind)+'</h4><span class="badge '+(j.status==="completed"?"teal":j.status==="error"?"red":"")+'">'+esc(j.status)+'</span></div><p>'+esc(j.message)+'</p><progress max="'+j.total+'" value="'+j.completed+'"></progress><div class="job-title"><small>'+j.completed+' / '+j.total+' cases · '+j.cases.filter(c=>c.error).length+' setup failures</small>'+(['queued','running','cancel_requested'].includes(j.status)?'<button data-cancel="'+j.id+'">Cancel remaining</button>':'')+'</div>'+j.cases.map(c=>{const r=state.catalog.find(r=>r.path===c.path);return'<div class="case-row"><span>'+esc(c.label)+' · return '+num(c.return)+' · unsafe '+c.unsafe_actions+' · '+esc(c.stop_reason)+(c.error?'<br><span class="text-red">'+esc(c.error)+'</span>':'')+'</span>'+(r?'<button data-open="'+r.id+'">Open replay ↗</button>':'')+'</div>';}).join("")+'</div>').join("")||'<p class="subtle">Your completed and queued experiments will appear here.</p>';
}
function csvDownload(){
  const rows=state.aggregates;if(!rows.length)return;
  const keys=Object.keys(rows[0]),cell=v=>'"'+String(v??"").replaceAll('"','""')+'"';
  const csv=[keys.map(cell).join(","),...rows.map(r=>keys.map(k=>cell(typeof r[k]==="object"?JSON.stringify(r[k]):r[k])).join(","))].join("\r\n");
  const url=URL.createObjectURL(new Blob([csv],{type:"text/csv"})),a=document.createElement("a");a.href=url;a.download="snomdp_map_level_comparison.csv";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
function downloadGraph(id){
  const svg=$(id).querySelector("svg");if(!svg)return;
  const copy=svg.cloneNode(true);copy.setAttribute("xmlns","http://www.w3.org/2000/svg");copy.setAttribute("font-family","Arial, sans-serif");
  const node=(name,attrs,text)=>{const e=document.createElementNS("http://www.w3.org/2000/svg",name);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,v);if(text)e.textContent=text;return e;};
  const legend=[...($(id).querySelector(".legend")?.children||[])],h=Number(copy.getAttribute("viewBox").split(" ")[3]),height=h+55+Math.ceil(legend.length/2)*20;
  copy.setAttribute("viewBox","0 0 1000 "+height);copy.setAttribute("width","1000");copy.setAttribute("height",height);copy.prepend(node("rect",{x:0,y:0,width:1000,height,fill:"white"}));
  const metadata=id==="tradeoff-chart"?state.analysisSnapshot:{created:new Date().toISOString(),metric:id==="replay-chart"?$("replay-metric").value:$("comparison-metric").value,runs:(id==="replay-chart"?[state.run.record]:state.catalog.filter(r=>state.selected.has(r.id))).map(r=>({id:r.id,path:r.path,config:r.config,map_hash:r.map_hash,model_hash:r.model_hash}))};
  copy.prepend(node("metadata",{},JSON.stringify(metadata)));
  const label=id==="replay-chart"?$("replay-metric").selectedOptions[0].text:id==="comparison-chart"?$("comparison-metric").selectedOptions[0].text:"Mean return versus mean unsafe actions";
  copy.append(node("text",{x:65,y:h+24,"font-size":13,fill:"#172e37"},label));
  legend.forEach((entry,i)=>{const x=65+(i%2)*465,y=h+48+Math.floor(i/2)*20;copy.append(node("rect",{x,y:y-7,width:16,height:3,fill:entry.querySelector("i").style.background}));copy.append(node("text",{x:x+24,y,"font-size":10,fill:"#465e67"},entry.textContent.slice(0,78)));});
  const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(copy)],{type:"image/svg+xml"})),a=document.createElement("a");a.href=url;a.download="snomdp_"+id+".svg";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
async function init(){
  const status=await api("/api/status");state.token=status.token;state.defaults=status.defaults;
  $("frozen-status").textContent=status.frozen.matches?"Scientific source matches frozen protocol.":status.frozen.available?"Scientific source differs from frozen protocol.":"No frozen protocol found.";
  const advanced={grid_spacing:"Grid spacing",gp_length_scale:"GP kernel length scale",gp_prior_variance:"GP prior variance",reward_confidence_multiplier:"Reward multiplier",exploration_accuracy:"Exploration accuracy ε",discount:"Discount γ",expander_lipschitz:"Expander Lipschitz L",value_tolerance:"Value iteration tolerance"};
  $("advanced-fields").innerHTML=Object.entries(advanced).map(([k,label])=>'<label>'+label+'<input name="'+k+'" type="number" step="any" required></label>').join("");
  $("preset-checks").innerHTML=[["source","ES2 retained · c=2"],["confidence","ES2 retained · c=4"],["revalidated","ES2 revalidated · c=4"],["oracle","True-field oracle"],["basic","Basic SNO-MDP"],["safemdp","SafeMDP"]].map(([v,t],i)=>'<label><input type="checkbox" name="comparison-preset" value="'+v+'" '+(i<4?"checked":"")+'> '+t+'</label>').join("");
  preset("smoke");
  const scale=document.createElement("div");scale.className="field-controls";scale.innerHTML='<label>Colour scale <select id="field-scale"><option value="run">Fixed for this run</option><option value="frame">Rescale per frame</option></select></label><span id="field-legend" class="field-legend"></span>';$("field-map").before(scale);
  const markers=document.createElement("div");markers.className="field-controls";markers.innerHTML='<span>□ Start &nbsp; <span style="color:#9f62b6">□ Target</span> &nbsp; ● Rover</span>';$("belief-map").before(markers);
  for(const id of ["replay-chart","comparison-chart","tradeoff-chart"]){const button=document.createElement("button");button.type="button";button.textContent="Export graph SVG ↓";button.style.fontSize="11px";button.onclick=()=>downloadGraph(id);$(id).after(button);}
  document.querySelectorAll(".nav").forEach(e=>e.onclick=()=>tab(e.dataset.tab));
  $("search").oninput=renderLibrary;$("group").onchange=renderLibrary;$("refresh").onclick=handle(refresh);
  document.body.addEventListener("click",handle(async e=>{
    const open=e.target.closest("[data-open]");if(open){await loadRun(open.dataset.open);return;}
    const remove=e.target.closest("[data-remove]");if(remove){state.selected.delete(remove.dataset.remove);renderLibrary();await drawSelected();return;}
    const cancel=e.target.closest("[data-cancel]");if(cancel){await api("/api/jobs/"+cancel.dataset.cancel+"/cancel",{});await pollJobs();return;}
    const pre=e.target.closest("[data-preset]");if(pre)preset(pre.dataset.preset);
  }));
  $("library").addEventListener("change",handle(async e=>{const id=e.target.dataset.select;if(!id)return;if(e.target.checked){if(state.selected.size>=12){e.target.checked=false;throw Error("Select at most 12 trajectories. Aggregate analysis can include the full frozen study.");}state.selected.add(id);}else state.selected.delete(id);$("selected-count").textContent=state.selected.size;if(state.tab==="compare")await drawSelected();}));
  $("prev").onclick=()=>setStep(state.step-1);$("next").onclick=()=>setStep(state.step+1);$("play").onclick=play;$("step").oninput=e=>setStep(Number(e.target.value));$("speed").onchange=()=>{if(state.timer){stop();play();}};
  const jump=i=>{stop();state.cell=null;setStep(i);};
  $("jump-start").onclick=()=>jump(0);$("jump-before").onclick=()=>jump(state.firstUnsafe-1);$("jump-unsafe").onclick=()=>jump(state.firstUnsafe);$("jump-switch").onclick=()=>jump(state.run.summary.transition_step);$("jump-final").onclick=()=>jump(state.run.frames.length-1);
  $("field-scale").onchange=()=>state.run&&drawMap("field-map",false);
  $("field").onchange=()=>state.run&&drawMap("field-map",false);$("arrows").onchange=()=>state.run&&drawMap("belief-map",true);$("replay-metric").onchange=()=>state.run&&renderStep();
  for(const id of ["belief-map","field-map"]){const canvas=$(id);canvas.onmousemove=e=>{if(!state.run)return;const cell=mapCell(canvas,e);if(!cell){hideTooltip();return;}const f=state.run.frames[state.step];tooltip(e,"Cell ("+cell.join(", ")+")\n"+(id==="belief-map"?(f.pessimistic[cell[0]][cell[1]]?"Executable region":f.optimistic[cell[0]][cell[1]]?"Optimistic only":"Uncertified"):$("field").selectedOptions[0].text+": "+num(fieldData()?.[cell[0]]?.[cell[1]],4))+"\nClick to inspect directed actions");};canvas.onmouseleave=hideTooltip;canvas.onclick=e=>{if(!state.run)return;state.cell=mapCell(canvas,e);renderStep();};}
  $("inspect-rover").onclick=()=>{state.cell=null;renderStep();};
  $("inspect-coordinate").onclick=handle(()=>{const x=Number($("cell-x").value),y=Number($("cell-y").value);if(!Number.isInteger(x)||!Number.isInteger(y)||x<0||y<0||x>=state.run.map.rows||y>=state.run.map.cols)throw Error("Choose valid integer cell coordinates.");state.cell=[x,y];renderStep();});
  $("comparison-metric").onchange=handle(drawSelected);$("analyse").onclick=handle(analyse);$("export-comparison").onclick=csvDownload;
  $("clear-selection").onclick=handle(async()=>{state.selected.clear();renderLibrary();await drawSelected();});
  $("matched-demo").onclick=handle(async()=>{const rows=state.catalog.filter(r=>r.group==="frozen"&&r.map_seed===100&&r.observation_seed===3000&&(r.method==="oracle"||(r.method==="sno_es2"&&r.noise_factor===1)));state.selected=new Set(rows.map(r=>r.id));renderLibrary();await drawSelected();});
  $("use-config").onclick=()=>{applyConfig(state.run.config);$("method").value=state.run.record.method;tab("experiment");};
  $("kind").onchange=updateExperimentMode;
  $("experiment-form").addEventListener("input",invalidatePlan);
  for(const id of ["aggregate-scope","aggregate-noise","aggregate-map","aggregate-observation","aggregate-batch"])$(id).addEventListener("change",invalidateAnalysis);
  $("export-analysis").onclick=()=>{if(!state.analysisSnapshot)return;const url=URL.createObjectURL(new Blob([JSON.stringify(state.analysisSnapshot,null,2)],{type:"application/json"})),a=document.createElement("a");a.href=url;a.download="snomdp_analysis_snapshot.json";a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
  $("preview").onclick=handle(preview);
  $("experiment-form").onsubmit=handle(async e=>{e.preventDefault();if(!$("experiment-form").reportValidity())return;clearError();$("start-job").disabled=true;try{await api("/api/jobs",request());await pollJobs();$("jobs").scrollIntoView({behavior:"smooth",block:"center"});}finally{$("start-job").disabled=false;}});
  $("load-config").onclick=()=>$("config-file").click();
  $("config-file").onchange=handle(async e=>{const file=e.target.files[0];if(!file)return;const c=JSON.parse(await file.text());await api("/api/plan",{kind:"single",config:c,method:$("method").value});applyConfig({...state.defaults,...c});});
  document.addEventListener("keydown",e=>{if(state.tab!=="replay"||["INPUT","SELECT","TEXTAREA","BUTTON"].includes(document.activeElement.tagName))return;if(e.key==="ArrowRight"){e.preventDefault();setStep(state.step+1);}if(e.key==="ArrowLeft"){e.preventDefault();setStep(state.step-1);}});
  await refresh();await pollJobs();
  const initial=state.catalog.find(r=>r.path==="results/demo_source")||state.catalog[0];if(initial)await loadRun(initial.id);
  setInterval(()=>pollJobs().catch(showError),2500);
}
init().catch(showError);
