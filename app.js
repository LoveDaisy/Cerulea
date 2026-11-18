import {vs, fs} from './shaders.js';
import {sha256} from './integrity.js';
const $=id=>document.getElementById(id), canvas=$('sky');
const FORMAT='cerulea-xyz-log8-v1', CPU_LIMIT=32, GPU_LIMIT=24, FETCH_LIMIT=4;
performance.setResourceTimingBufferSize(5000);
let gl,program,axisTexture,manifest,active,requested,epoch=0,serial=0,pumping=false,lost=false,disposed=false,raf=0;
const cpu=new Map(),gpu=new Map(),waiters=new Map(),flightControllers=new Set(),listeners=new AbortController();
const stats={loadedBytes:0,cpuBytes:0,deletedTextures:0,createdTextures:0,inflight:0,maxInflight:0,commits:0,stale:0};
let sampleSize=0, fixedExposure=1;
const loc=n=>gl.getUniformLocation(program,n);
function on(target,type,fn,options={}){target.addEventListener(type,fn,{...options,signal:listeners.signal});}
function labels(){for(const id of ['height','aod','yaw','pitch','fov','ev'])$(id+'Value').textContent=Number($(id).value).toFixed(id==='aod'?4:id==='height'?2:id==='ev'?1:0)+(id==='aod'?'':id==='ev'?' EV':'°');}
function stateText(state){return `太阳 ${state[2].toFixed(2)}°，AOD ${state[0].toFixed(4)}，臭氧 ${state[1]} DU`;}
function fail(error){$('status').textContent=`无法完成加载：${error.message}。当前画面参数见画面左上；请重试或检查浏览器能力与静态文件。`;$('retry').hidden=false;}
function bracket(a,x){let i=0;while(i<a.length-2&&a[i+1]<=x)i++;return i;}
function resolve(state){
  const [ad,du,h]=state;
  if(!Number.isFinite(ad)||!Number.isFinite(h)||ad<0||ad>.3||h< -6||h>90||![0,300].includes(du))throw Error('参数超出声明域');
  const hs=manifest.heights,as=manifest.aods,ih=bracket(hs,h),ia=bracket(as,ad),start=Math.max(0,Math.min(as.length-4,ia-1)),nodes=as.slice(start,start+4);
  const scenes=nodes.flatMap(a=>[hs[ih],hs[ih+1]].map(v=>manifest.scenes.find(s=>s.state[0]===a&&s.state[1]===du&&s.state[2]===v)));
  if(scenes.some(s=>!s))throw Error('清单缺少场景');
  return {state:[...state],scenes,heights:[hs[ih],hs[ih+1]],weight:(h-hs[ih])/(hs[ih+1]-hs[ih]),nodes,interval:ia-start};
}
function texture(data,w,h,unit=0,byteOutput=false){
  const t=gl.createTexture();if(!t)throw Error('纹理分配失败');
  gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);
  for(const [p,v] of [[gl.TEXTURE_MIN_FILTER,gl.NEAREST],[gl.TEXTURE_MAG_FILTER,gl.NEAREST],[gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE],[gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE]])gl.texParameteri(gl.TEXTURE_2D,p,v);
  gl.texImage2D(gl.TEXTURE_2D,0,byteOutput?gl.RGBA8:gl.RGBA32F,w,h,0,gl.RGBA,byteOutput?gl.UNSIGNED_BYTE:gl.FLOAT,data);return t;
}
function deleteTexture(t){gl.deleteTexture(t);stats.deletedTextures++;}
function initGL(){
  gl=canvas.getContext('webgl2',{preserveDrawingBuffer:true,antialias:false});
  if(!gl)throw Error('浏览器未提供 WebGL2，请启用硬件加速或更换浏览器');
  program=gl.createProgram();
  for(const [type,source] of [[gl.VERTEX_SHADER,vs],[gl.FRAGMENT_SHADER,fs]]){
    const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);
    if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));
    gl.attachShader(program,s);gl.deleteShader(s);
  }
  gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
  const arrays=[manifest.mu,manifest.azimuth,manifest.psi,manifest.chi];
  const width=Math.max(...arrays.map(a=>a.length));const data=new Float32Array(width*4*4);
  arrays.forEach((a,row)=>a.forEach((v,i)=>data[(row*width+i)*4]=v));axisTexture=texture(data,width,4,8);
  gl.uniform4i(loc('counts'),...arrays.map(a=>a.length));
}
function validateManifest(m){
  if(m.format!==FORMAT||m.display!=='core-aces-srgb-v1'||m.reconstruction!=='xyz-cap-coverage-v1')throw Error('数据／显示版本不匹配');
  const axes=[m.mu,m.azimuth,m.psi,m.chi,m.heights,m.aods];
  const endpoints=[[0,1],[0,180],[0,12],[0,360],[-6,90],[0,.3]];
  axes.forEach((a,i)=>{if(!Array.isArray(a)||a.length<2||a.length>256||a[0]!==endpoints[i][0]||a.at(-1)!==endpoints[i][1]||a.some((x,j)=>!Number.isFinite(x)||(j&&x<=a[j-1])))throw Error('清单网格无效');});
  if(m.aods.length<4||m.scenes.length!==m.heights.length*m.aods.length*2)throw Error('清单人口不完整');
  const keys=new Set(),files=new Set();
  for(const s of m.scenes){
    if(!Array.isArray(s.state)||s.state.length!==3||!m.aods.includes(s.state[0])||![0,300].includes(s.state[1])||!m.heights.includes(s.state[2])||keys.has(s.state.join(','))||files.has(s.file)||!/^assets\/[a-zA-Z0-9.\-]+\.bin$/.test(s.file)||!Number.isFinite(s.base_exposure)||s.base_exposure<=0||!Number.isInteger(s.bytes)||s.bytes<25||s.bytes>1e6||!/^[0-9a-f]{64}$/.test(s.sha256))throw Error('清单场景无效');
    keys.add(s.state.join(','));files.add(s.file);
  }
}
function touch(map,key){const v=map.get(key);map.delete(key);map.set(key,v);return v;}
function trimCPU(){while(cpu.size>CPU_LIMIT){const [key,v]=cpu.entries().next().value;stats.cpuBytes-=v.data.byteLength;cpu.delete(key);}}
async function load(scene){
  if(cpu.has(scene.file))return touch(cpu,scene.file);
  stats.inflight++;stats.maxInflight=Math.max(stats.maxInflight,stats.inflight);
  const controller=new AbortController();flightControllers.add(controller);const timeout=setTimeout(()=>controller.abort(),15000);
  try{
    const r=await fetch(scene.file,{cache:'no-store',signal:controller.signal});if(!r.ok)throw Error(`天空资产 HTTP ${r.status}`);
    const bytes=await r.arrayBuffer();stats.loadedBytes+=bytes.byteLength;
    if(bytes.byteLength!==scene.bytes||await sha256(bytes)!==scene.sha256)throw Error('天空资产校验失败，请重新部署完整数据');
    const header=new Float32Array(bytes.slice(0,24));if(![...header].every(Number.isFinite)||[...header.slice(3)].some(x=>x<=0))throw Error('编码头无效');
    const raw=new Uint8Array(await new Response(new Blob([bytes.slice(24)]).stream().pipeThrough(new DecompressionStream('deflate'))).arrayBuffer());
    const {mu,azimuth,psi,chi}=manifest,nw=mu.length*azimuth.length,nc=psi.length*chi.length;
    if(raw.length!==(nw+nc)*3)throw Error('解压后长度无效');
    const width=Math.max(azimuth.length,chi.length),height=mu.length+psi.length,data=new Float32Array(width*height*4);
    for(let i=0;i<nw+nc;i++){
      let row,col,alpha=1;
      if(i<nw){row=Math.floor(i/azimuth.length);col=i%azimuth.length;}
      else {const k=i-nw;row=mu.length+Math.floor(k/chi.length);col=k%chi.length;const p=psi[Math.floor(k/chi.length)]*Math.PI/180,c=chi[col]*Math.PI/180,h=scene.state[2]*Math.PI/180;alpha=Math.cos(p)*Math.sin(h)+Math.sin(p)*Math.cos(c)*Math.cos(h);}
      const k=(row*width+col)*4;for(let j=0;j<3;j++)data[k+j]=raw[i*3+j]===0?0:2**(header[j]+(raw[i*3+j]-1)*header[j+3]);data[k+3]=alpha;
    }
    if(!data.every(Number.isFinite))throw Error('解码出现非有限值');
    const value={scene,data,width,height};if(!disposed){cpu.set(scene.file,value);stats.cpuBytes+=data.byteLength;trimCPU();}return value;
  }finally{clearTimeout(timeout);flightControllers.delete(controller);stats.inflight--;}
}
function upload(v){if(gpu.has(v.scene.file))return touch(gpu,v.scene.file);const tex=texture(v.data,v.width,v.height);stats.createdTextures++;const obj={scene:v.scene,tex};gpu.set(v.scene.file,obj);return obj;}
function trimGPU(){const keep=new Set(active?.scenes.map(s=>s.file));for(const [key,v] of gpu){if(gpu.size<=GPU_LIMIT)break;if(!keep.has(key)){deleteTexture(v.tex);gpu.delete(key);}}}
function exposure(){if($('exposureMode').value==='fixed')return fixedExposure*2**Number($('ev').value);
  const a=active,i=a.interval*2,t=(a.state[0]-a.nodes[a.interval])/(a.nodes[a.interval+1]-a.nodes[a.interval]);
  const e=j=>Math.log(a.loaded[j].scene.base_exposure)*(1-a.weight)+Math.log(a.loaded[j+1].scene.base_exposure)*a.weight;
  return Math.exp((1-t)*e(i)+t*e(i+2))*2**Number($('ev').value);
}
function draw(){
  if(!active||lost||disposed||document.hidden)return;
  const a=active;gl.useProgram(program);gl.viewport(0,0,sampleSize||canvas.width,sampleSize?1:canvas.height);
  a.loaded.forEach((v,i)=>{gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,v.tex);gl.uniform1i(loc('atlas'+i),i);});
  gl.activeTexture(gl.TEXTURE8);gl.bindTexture(gl.TEXTURE_2D,axisTexture);gl.uniform1i(loc('axes'),8);
  gl.uniform2f(loc('resolution'),canvas.width,canvas.height);gl.uniform2f(loc('heights'),...a.heights);gl.uniform4f(loc('aodNodes'),...a.nodes);gl.uniform1i(loc('aodInterval'),a.interval);gl.uniform1f(loc('heightWeight'),a.weight);gl.uniform1f(loc('targetHeight'),a.state[2]);gl.uniform1f(loc('aodValue'),a.state[0]);gl.uniform1f(loc('exposure'),exposure());
  for(const id of ['yaw','pitch','fov'])gl.uniform1f(loc(id),Number($(id).value));
  gl.uniform1i(loc('heightMode'),a.heights[1]<=2?1:0);gl.uniform1i(loc('pchip'),1);gl.uniform1i(loc('transportCap'),1);gl.drawArrays(gl.TRIANGLES,0,3);
}
function schedule(){if(!raf)raf=requestAnimationFrame(()=>{raf=0;draw();});labels();}
function resize(){const dpr=Math.min(devicePixelRatio||1,1.5),scale=Math.min(1,1280/(canvas.clientWidth*dpr));canvas.width=Math.max(1,Math.round(canvas.clientWidth*dpr*scale));canvas.height=Math.max(1,Math.round(canvas.clientHeight*dpr*scale));schedule();}
async function pump(){
  if(pumping||lost||disposed)return;pumping=true;
  try{while(requested&&!lost&&!disposed){
    const req=requested;requested=null;const version=epoch;
    try{
      const a=resolve(req.state),data=[];
      for(let i=0;i<a.scenes.length;i+=FETCH_LIMIT){
        // allSettled avoids leaving unaccounted sibling fetches after one failure.
        const batch=await Promise.allSettled(a.scenes.slice(i,i+FETCH_LIMIT).map(load));
        const bad=batch.find(r=>r.status==='rejected');if(bad)throw bad.reason;
        data.push(...batch.map(r=>r.value));
        if(req.id!==serial||version!==epoch)break;
      }
      if(req.id!==serial||version!==epoch||lost||disposed){stats.stale++;waiters.get(req.id)?.resolve(false);continue;}
      const previous=active;
      const keep=new Set([...(active?.scenes??[]),...a.scenes].map(s=>s.file));
      const needed=a.scenes.filter(s=>!gpu.has(s.file)).length;
      for(const [key,v] of gpu){if(gpu.size+needed<=GPU_LIMIT)break;if(!keep.has(key)){deleteTexture(v.tex);gpu.delete(key);}}
      try{active={...a,loaded:data.map(upload),id:req.id};draw();
        if(gl.getError()!==gl.NO_ERROR)throw Error('GPU 绘制失败');
      }catch(e){active=previous;draw();throw e;}finally{trimGPU();}
      stats.commits++;$('rendered').textContent='当前画面：'+stateText(active.state);
      $('status').textContent='已呈现。拖动天空探索其他方向。';$('retry').hidden=true;
      waiters.get(req.id)?.resolve(true);
    }catch(e){if(req.id===serial&&!lost&&!disposed){fail(e);waiters.get(req.id)?.reject(e);}else waiters.get(req.id)?.resolve(false);}
    finally{waiters.delete(req.id);}
  }}finally{pumping=false;if(requested&&!lost&&!disposed)void pump();}
}
function setState(state){
  if(!manifest||disposed)return Promise.reject(Error('取景器尚未就绪'));
  try{resolve(state);}catch(e){return Promise.reject(e);}
  for(const [id,w] of waiters){w.resolve(false);waiters.delete(id);}
  for(const controller of flightControllers)controller.abort();
  const id=++serial;requested={id,state:[...state]};
  $('status').textContent=`正在加载：${stateText(state)}。${active?'当前仍显示左上所列场景。':'尚未呈现。'}`;$('retry').hidden=true;
  const promise=new Promise((resolve,reject)=>waiters.set(id,{resolve,reject}));void pump();return promise;
}
function update(){return setState([Number($('aod').value),Number($('ozone').value),Number($('height').value)]);}
function sample(directions,displayOutput=false,overrideExposure=null){
  if(!active||lost)throw Error('无可采样场景');
  if(!displayOutput&&!gl.getExtension('EXT_color_buffer_float'))throw Error('XYZ 审计需要 EXT_color_buffer_float');
  const n=directions.length;if(!n||n>gl.getParameter(gl.MAX_TEXTURE_SIZE))throw Error('采样数量无效');
  const pts=new Float32Array(n*4);directions.forEach((v,i)=>pts.set(v,i*4));let dt,target,fb;
  try{dt=texture(pts,n,1,9);target=texture(null,n,1,10,displayOutput);fb=gl.createFramebuffer();gl.bindFramebuffer(gl.FRAMEBUFFER,fb);gl.framebufferTexture2D(gl.FRAMEBUFFER,gl.COLOR_ATTACHMENT0,gl.TEXTURE_2D,target,0);
    if(gl.checkFramebufferStatus(gl.FRAMEBUFFER)!==gl.FRAMEBUFFER_COMPLETE)throw Error('审计帧缓冲不可用');
    gl.uniform1i(loc('probeDirections'),9);gl.uniform1i(loc('probing'),1);gl.uniform1i(loc('floatOutput'),displayOutput?0:1);sampleSize=n;draw();
    if(overrideExposure!==null){gl.uniform1f(loc('exposure'),overrideExposure);gl.drawArrays(gl.TRIANGLES,0,3);}
    const pixels=displayOutput?new Uint8Array(n*4):new Float32Array(n*4);gl.readPixels(0,0,n,1,gl.RGBA,displayOutput?gl.UNSIGNED_BYTE:gl.FLOAT,pixels);
    if(gl.getError()!==gl.NO_ERROR)throw Error('GPU 回读失败');
    return Array.from({length:n},(_,i)=>Array.from(pixels.slice(i*4,i*4+3),v=>displayOutput?v/255:v));
  }finally{gl.bindFramebuffer(gl.FRAMEBUFFER,null);gl.uniform1i(loc('probing'),0);gl.uniform1i(loc('floatOutput'),0);if(dt)gl.deleteTexture(dt);if(target)gl.deleteTexture(target);if(fb)gl.deleteFramebuffer(fb);sampleSize=0;draw();}
}
function dispose(){disposed=true;epoch++;serial++;requested=null;for(const c of flightControllers)c.abort();for(const w of waiters.values())w.resolve(false);waiters.clear();cancelAnimationFrame(raf);listeners.abort();if(gl&&!lost){for(const v of gpu.values())deleteTexture(v.tex);gl.deleteTexture(axisTexture);gl.deleteProgram(program);}gpu.clear();cpu.clear();stats.cpuBytes=0;active=null;}
on(window,'resize',resize);on(document,'visibilitychange',()=>{if(!document.hidden)schedule();});
on(window,'pagehide',e=>{if(!e.persisted)dispose();});
on(window,'pageshow',e=>{if(e.persisted){resize();if(!active&&!lost&&manifest)update().catch(fail);}});
on(canvas,'webglcontextlost',e=>{e.preventDefault();lost=true;epoch++;for(const c of flightControllers)c.abort();active=null;gpu.clear();$('rendered').textContent='GPU 上下文已丢失，当前无有效场景';$('status').textContent='正在等待 GPU 恢复，恢复后重新加载当前输入。';});
on(canvas,'webglcontextrestored',()=>{try{lost=false;initGL();resize();update().catch(fail);}catch(e){fail(e);}});
for(const id of ['height','aod','ozone'])on($(id),'input',()=>{labels();update().catch(fail);});
for(const id of ['ev','yaw','pitch','fov','exposureMode'])on($(id),'input',schedule);
on($('retry'),'click',()=>{if(manifest&&program&&axisTexture&&!lost&&!disposed)update().catch(fail);else location.reload();});
on($('faceSun'),'click',()=>{$('yaw').value=0;$('pitch').value=Math.max(0,active?.state[2]??3);schedule();});
on($('fullscreen'),'click',()=>{if(canvas.requestFullscreen)canvas.requestFullscreen().catch(fail);else fail(Error('此浏览器不支持画布全屏，可使用系统浏览器全屏'));});
let drag;
on(canvas,'pointerdown',e=>{drag=[e.clientX,e.clientY,Number($('yaw').value),Number($('pitch').value)];canvas.setPointerCapture(e.pointerId);});
on(canvas,'pointermove',e=>{if(drag){$('yaw').value=((drag[2]-(e.clientX-drag[0])*.15+540)%360)-180;$('pitch').value=Math.max(-20,Math.min(90,drag[3]+(e.clientY-drag[1])*.15));schedule();}});
for(const name of ['pointerup','pointercancel'])on(canvas,name,()=>{drag=null;});
on(canvas,'wheel',e=>{e.preventDefault();$('fov').value=Math.max(30,Math.min(120,Number($('fov').value)+e.deltaY*.03));schedule();},{passive:false});
on(canvas,'keydown',e=>{const actions={ArrowLeft:['yaw',-3],ArrowRight:['yaw',3],ArrowUp:['pitch',3],ArrowDown:['pitch',-3],'+':['fov',-3],'-':['fov',3],'=':['fov',-3]};if(actions[e.key]){e.preventDefault();const [id,delta]=actions[e.key],el=$(id);el.value=Math.max(Number(el.min),Math.min(Number(el.max),Number(el.value)+delta));schedule();}});
window.demo={setState(state){try{resolve(state);$('aod').value=state[0];$('ozone').value=state[1];$('height').value=state[2];labels();return update();}catch(e){return Promise.reject(e);}},sample,draw,resize,dispose,
  get gl(){return gl;},get canvas(){return canvas;},get active(){return active?{state:[...active.state],id:active.id}:null;},get manifest(){return manifest;},get exposure(){return active?exposure():null;},
  get stats(){return {...stats,cpuEntries:cpu.size,gpuEntries:gpu.size,cpuLimit:CPU_LIMIT,gpuLimit:GPU_LIMIT,fetchLimit:FETCH_LIMIT,pending:waiters.size,lost,disposed};},
  clearCache(){for(const v of gpu.values())deleteTexture(v.tex);gpu.clear();cpu.clear();stats.cpuBytes=0;active=null;},validateManifest};
(async()=>{try{
  if(typeof DecompressionStream==='undefined')throw Error('浏览器缺少 DecompressionStream，请升级到支持原生 deflate 解压的浏览器');
  const response=await fetch('manifest.json',{cache:'no-store'});if(!response.ok)throw Error(`清单 HTTP ${response.status}`);manifest=await response.json();validateManifest(manifest);initGL();resize();labels();
  $('dataInfo').textContent=`数据 ${manifest.format}；核心 ${manifest.source.core_sha256.slice(0,16)}；${manifest.scenes.length} 场，天空资产 ${(manifest.scenes.reduce((s,r)=>s+r.bytes,0)/1e6).toFixed(2)} MB。数据是来自修复后核心的新求解。`;
  await update();window.firstReadyMs=performance.now();
}catch(e){fail(e);}})();
