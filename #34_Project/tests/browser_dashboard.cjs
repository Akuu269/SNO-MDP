/* Optional browser regression: start python3 -m analysis_ui first.
 * Node 18+ and Chrome are needed only for this developer check.
 * Creates one small matched batch under results/interactive/.
 */
"use strict";
const assert = require("node:assert/strict");
const {spawn} = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const profile = fs.mkdtempSync(path.join(os.tmpdir(), "snomdp-ui-test-"));
const chrome = spawn(process.env.CHROME_BIN || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ["--headless=new", "--remote-debugging-pipe", "--no-first-run", "--no-default-browser-check",
    "--disable-gpu", "--user-data-dir=" + profile], {stdio:["ignore","ignore","pipe","pipe","pipe"]});
let next=0,buffer="",session;const pending=new Map(),errors=[];
chrome.stdio[4].on("data",b=>{buffer+=b.toString();let i;while((i=buffer.indexOf("\0"))>=0){
  const msg=JSON.parse(buffer.slice(0,i));buffer=buffer.slice(i+1);
  if(msg.id){const p=pending.get(msg.id);pending.delete(msg.id);if(p)msg.error?p.reject(Error(JSON.stringify(msg.error))):p.resolve(msg.result);}
  if(msg.method==="Runtime.exceptionThrown")errors.push(msg.params.exceptionDetails);
}});
chrome.on("error",e=>{console.error(e.message);process.exitCode=1;});
function command(method,params={},sid=session){return new Promise((resolve,reject)=>{
  const id=++next;pending.set(id,{resolve,reject});
  chrome.stdio[3].write(JSON.stringify({id,method,params,...(sid?{sessionId:sid}:{})})+"\0");
});}
async function evaluate(expression){const r=await command("Runtime.evaluate",{expression,awaitPromise:true,returnByValue:true});
  if(r.exceptionDetails)throw Error(JSON.stringify(r.exceptionDetails));return r.result.value;}
async function until(expression,limit=30000){const end=Date.now()+limit;while(Date.now()<end){
  const v=await evaluate(expression);if(v)return v;await new Promise(r=>setTimeout(r,100));
}throw Error("Timed out: "+expression);}
const click = id => evaluate('document.getElementById('+JSON.stringify(id)+').click()');
const timeout=setTimeout(()=>{console.error("Browser test timeout");chrome.kill();process.exit(1);},120000);
(async()=>{try{
  const {targetId}=await command("Target.createTarget",{url:"about:blank"},null);
  session=(await command("Target.attachToTarget",{targetId,flatten:true},null)).sessionId;
  await command("Runtime.enable");await command("Page.enable");
  await command("Emulation.setDeviceMetricsOverride",{width:1560,height:1100,deviceScaleFactor:1,mobile:false});
  await command("Page.navigate",{url:process.env.LAB_URL||"http://127.0.0.1:8765"});
  await until('document.querySelectorAll("#inspector tbody tr").length===5');
  await evaluate('document.getElementById("field").value="altitude_std";document.getElementById("field").dispatchEvent(new Event("change"))');
  const scale=await evaluate('document.getElementById("field-caption").textContent');
  await click("jump-final");assert.equal(await evaluate('document.getElementById("field-caption").textContent'),scale);
  await click("jump-before");assert.equal(await evaluate('document.getElementById("step").value'),"96");
  await until('document.getElementById("inspector-title").textContent.includes("step 96")');
  console.log("BEFORE_VIOLATION",await evaluate('document.getElementById("inspector").innerText'));
  await click("jump-unsafe");assert.equal(await evaluate('document.getElementById("step").value'),"97");
  assert.match(await evaluate('document.getElementById("step-evidence").textContent'),/violated the constraint/);
  const shot=await command("Page.captureScreenshot",{format:"png",captureBeyondViewport:false});
  fs.writeFileSync(path.join(os.tmpdir(),"snomdp-ui-reviewed.png"),Buffer.from(shot.data,"base64"));
  await click("jump-switch");
  assert.equal(await evaluate('document.getElementById("phase").textContent'),"reward optimization");
  await evaluate('document.querySelector("[data-tab=compare]").click()');await click("matched-demo");
  await until('document.querySelectorAll("#comparison-chart path").length===4 && document.querySelectorAll("#aggregate-table tbody tr").length===6');
  assert.match(await evaluate('document.getElementById("aggregate-summary").textContent'),/72 unique/);
  await evaluate('document.getElementById("aggregate-noise").value="2";document.getElementById("aggregate-noise").dispatchEvent(new Event("change"))');
  assert.equal(await evaluate('document.querySelector("#tradeoff-chart svg")===null && document.getElementById("export-comparison").disabled'),true);
  await click("analyse");await until('document.getElementById("aggregate-summary").textContent.includes("noise factor: 2")');
  await evaluate('document.querySelector("[data-tab=experiment]").click();document.getElementById("method").value="oracle";document.querySelector("[data-preset=source]").click()');
  assert.equal(await evaluate('document.getElementById("method").value'),"sno_es2");
  await evaluate('document.querySelector("[data-preset=smoke]").click();document.querySelector("[name=horizon]").value=4;document.querySelector("[name=discount]").value=.5');
  await click("preview");await until('document.getElementById("plan-count").textContent==="1 case planned"');
  await evaluate('document.querySelector("[name=horizon]").value=5;document.querySelector("[name=horizon]").dispatchEvent(new Event("input",{bubbles:true}))');
  assert.equal(await evaluate('document.getElementById("plan-detail").hidden && document.getElementById("plan-count").textContent===""'),true);
  await evaluate('document.querySelector("[name=horizon]").value=4;document.getElementById("kind").value="matched";document.getElementById("kind").dispatchEvent(new Event("change"));document.querySelectorAll("[name=comparison-preset]").forEach(e=>e.checked=["source","revalidated"].includes(e.value))');
  assert.equal(await evaluate('document.getElementById("method").disabled && document.querySelector("[name=safety_confidence_multiplier]").disabled'),true);
  await click("preview");await until('document.getElementById("plan-count").textContent==="2 cases planned"');
  const old=await evaluate('document.querySelector("#jobs h4")?.textContent');
  await click("start-job");
  await until('document.querySelector("#jobs h4")?.textContent!=='+JSON.stringify(old)+' && document.querySelector("#jobs .badge")?.textContent==="completed"');
  assert.match(await evaluate('document.querySelector("#jobs .job").innerText'),/2 \/ 2 cases · 0 setup failures/);
  await evaluate('document.querySelector("[data-tab=compare]").click();document.getElementById("aggregate-scope").value="interactive";document.getElementById("aggregate-noise").value="all";document.getElementById("aggregate-batch").value=document.querySelector("#jobs h4").textContent.split(" · ")[0];document.getElementById("aggregate-batch").dispatchEvent(new Event("change"))');
  await click("analyse");await until('document.getElementById("aggregate-summary").textContent.includes("2 unique")');
  assert.equal(await evaluate('document.querySelectorAll("#aggregate-table tbody tr").length'),2);
  const snapshot=await evaluate('(async()=>{let blob;const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;try{URL.createObjectURL=b=>{blob=b;return "blob:test"};HTMLAnchorElement.prototype.click=function(){};document.getElementById("export-analysis").click();return JSON.parse(await blob.text());}finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;}})()');
  assert.equal(snapshot.runs.length,2);assert.notEqual(snapshot.filters.batch,"all");assert.ok(snapshot.analysis.warnings.some(w=>w.includes("Multiple controls differ")));
  const exported=await evaluate('(async()=>{let blob;const create=URL.createObjectURL,click=HTMLAnchorElement.prototype.click;try{URL.createObjectURL=b=>{blob=b;return "blob:test"};HTMLAnchorElement.prototype.click=function(){};document.querySelector("#tradeoff-chart + button").click();const d=new DOMParser().parseFromString(await blob.text(),"image/svg+xml");return {valid:!d.querySelector("parsererror"),snapshot:JSON.parse(d.querySelector("metadata").textContent)};}finally{URL.createObjectURL=create;HTMLAnchorElement.prototype.click=click;}})()');
  assert.equal(exported.valid,true);assert.equal(exported.snapshot.filters.batch,snapshot.filters.batch);
  await evaluate('document.querySelector("[data-tab=experiment]").click()');
  await evaluate('document.querySelector("#jobs [data-open]").click()');await until('document.getElementById("step").max==="4" && !document.getElementById("tab-replay").hidden');
  await until('document.querySelectorAll("#inspector tbody tr").length===5');
  await command("Emulation.setDeviceMetricsOverride",{width:430,height:932,deviceScaleFactor:1,mobile:true});
  assert.equal(await evaluate('document.documentElement.scrollWidth<=innerWidth'),true);
  assert.equal(await evaluate('document.getElementById("notice").hidden'),true);
  assert.deepEqual(errors,[]);
  console.log("PASS: stable colour scales, event bookmarks, exact bounds, filter invalidation, batch isolation, snapshot/SVG provenance, presets, preview invalidation, effective controls, matched jobs, replay, and mobile layout.");
}catch(e){console.error(e);process.exitCode=1;}
finally{clearTimeout(timeout);try{await command("Browser.close",{},null);}catch{}chrome.kill();}})();
