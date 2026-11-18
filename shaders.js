export const vs=`#version 300 es
void main(){vec2 p=vec2((gl_VertexID<<1)&2,gl_VertexID&2);gl_Position=vec4(p*2.-1.,0,1);}`;
export const fs=`#version 300 es
precision highp float;
uniform sampler2D atlas0,atlas1,atlas2,atlas3,atlas4,atlas5,atlas6,atlas7,axes,probeDirections;
uniform ivec4 counts;
uniform vec2 resolution,heights;
uniform vec4 aodNodes;
uniform float yaw,pitch,fov,exposure,heightWeight,aodValue,targetHeight;
uniform int heightMode,aodInterval;
uniform bool probing,floatOutput,pchip,transportCap;
out vec4 frag;
const float PI=3.141592653589793;
float axis(int row,int i){return texelFetch(axes,ivec2(i,row),0).r;}
vec2 bracket(int row,int n,float x){int lo=0,hi=n-1;for(int k=0;k<8;k++){int m=(lo+hi)/2;if(axis(row,m)<=x)lo=m;else hi=m;}lo=min(lo,n-2);float a=axis(row,lo),b=axis(row,lo+1);return vec2(float(lo),clamp((x-a)/(b-a),0.,1.));}
vec4 bilinear(sampler2D tex,vec2 a,vec2 b,int offset){ivec2 ij=ivec2(int(b.x),int(a.x)+offset);return mix(mix(texelFetch(tex,ij,0),texelFetch(tex,ij+ivec2(1,0),0),b.y),mix(texelFetch(tex,ij+ivec2(0,1),0),texelFetch(tex,ij+ivec2(1,1),0),b.y),a.y);}
// Azimuth is arbitrary at a chart pole; GLSL atan(0,0) is undefined.
vec3 sampleSky(sampler2D tex,vec3 d,float endpoint){float az=(d.x==0.&&d.y==0.)?0.:abs(degrees(atan(d.y,d.x)));vec3 w=bilinear(tex,bracket(0,counts.x,d.z),bracket(1,counts.y,az),0).rgb;
float h=transportCap&&heights.x>=12.?targetHeight:endpoint;float r=radians(h);vec3 s=vec3(cos(r),0,sin(r)),e=vec3(-sin(r),0,cos(r));float psi=degrees(atan(length(vec2(dot(d,e),d.y)),dot(d,s)));if(psi>12.)return w;
float pe=dot(d,e);float chi=(d.y==0.&&pe==0.)?0.:mod(degrees(atan(d.y,pe))+360.,360.);vec2 a=bracket(2,counts.z,psi),b=bracket(3,counts.w,chi);ivec2 ij=ivec2(int(b.x),int(a.x)+counts.x);
// Interpolated corner confidence gives identical edge values in adjacent cells.
float coverage=0.;vec3 weighted=vec3(0.);for(int i=0;i<2;i++)for(int j=0;j<2;j++){vec4 v=texelFetch(tex,ij+ivec2(j,i),0);float weight=(i==0?1.-a.y:a.y)*(j==0?1.-b.y:b.y)*smoothstep(0.,sin(radians(.05)),v.a);coverage+=weight;weighted+=weight*v.rgb;}
float blend=smoothstep(0.,1.,(12.-psi)/4.)*smoothstep(0.,1.,coverage);return mix(w,weighted/max(coverage,1e-30),blend);}
vec3 heightMix(sampler2D a,sampler2D b,vec3 d){vec3 x=sampleSky(a,d,heights.x),y=sampleSky(b,d,heights.y);return heightMode==1?exp(mix(log(max(x,vec3(1e-30))),log(max(y,vec3(1e-30))),heightWeight)):mix(x,y,heightWeight);}
float harmonic(float d0,float d1,float h0,float h1){if(d0*d1<=0.)return 0.;float w1=2.*h1+h0,w2=h1+2.*h0;return (w1+w2)/(w1/d0+w2/d1);}
vec3 harmonic3(vec3 a,vec3 b,float h0,float h1){return vec3(harmonic(a.x,b.x,h0,h1),harmonic(a.y,b.y,h0,h1),harmonic(a.z,b.z,h0,h1));}
float endpoint(float d0,float d1,float h0,float h1){float m=((2.*h0+h1)*d0-h0*d1)/(h0+h1);if(m*d0<=0.)return 0.;if(d0*d1<0.&&abs(m)>abs(3.*d0))return 3.*d0;return m;}
vec3 endpoint3(vec3 a,vec3 b,float h0,float h1){return vec3(endpoint(a.x,b.x,h0,h1),endpoint(a.y,b.y,h0,h1),endpoint(a.z,b.z,h0,h1));}
vec3 aodMix(vec3 v0,vec3 v1,vec3 v2,vec3 v3){v0=log(max(v0,vec3(1e-30)));v1=log(max(v1,vec3(1e-30)));v2=log(max(v2,vec3(1e-30)));v3=log(max(v3,vec3(1e-30)));
float h0=aodNodes.y-aodNodes.x,h1=aodNodes.z-aodNodes.y,h2=aodNodes.w-aodNodes.z;vec3 d0=(v1-v0)/h0,d1=(v2-v1)/h1,d2=(v3-v2)/h2;
vec3 a,b,ma,mb;float h,x;if(aodInterval==0){a=v0;b=v1;ma=endpoint3(d0,d1,h0,h1);mb=harmonic3(d0,d1,h0,h1);h=h0;x=aodNodes.x;}else if(aodInterval==1){a=v1;b=v2;ma=harmonic3(d0,d1,h0,h1);mb=harmonic3(d1,d2,h1,h2);h=h1;x=aodNodes.y;}else{a=v2;b=v3;ma=harmonic3(d1,d2,h1,h2);mb=endpoint3(d2,d1,h2,h1);h=h2;x=aodNodes.z;}
float t=clamp((aodValue-x)/h,0.,1.);if(!pchip)return exp(mix(a,b,t));float t2=t*t,t3=t2*t;return exp((2.*t3-3.*t2+1.)*a+(t3-2.*t2+t)*h*ma+(-2.*t3+3.*t2)*b+(t3-t2)*h*mb);}
float aces(float x){x=max(x,0.)*.6;return clamp((x*(2.51*x+.03))/(x*(2.43*x+.59)+.14),0.,1.);}
void main(){vec2 p=(gl_FragCoord.xy/resolution*2.-1.)*tan(radians(fov)*.5);p.y*=resolution.y/resolution.x;float y=radians(yaw),t=radians(pitch);vec3 c=vec3(cos(t)*cos(y),cos(t)*sin(y),sin(t)),r=vec3(-sin(y),cos(y),0),u=vec3(-sin(t)*cos(y),-sin(t)*sin(y),cos(t));vec3 d=probing?texelFetch(probeDirections,ivec2(int(gl_FragCoord.x),0),0).rgb:normalize(c+r*p.x+u*p.y);if(d.z<0.){frag=vec4(vec3(.055),1);return;}
vec3 v0=heightMix(atlas0,atlas1,d),v1=heightMix(atlas2,atlas3,d),v2=heightMix(atlas4,atlas5,d),v3=heightMix(atlas6,atlas7,d);vec3 xyz=aodMix(v0,v1,v2,v3);if(floatOutput){frag=vec4(xyz,1);return;}
vec3 rgb=vec3(dot(vec3(3.2406,-1.5372,-.4986),xyz),dot(vec3(-.9689,1.8758,.0415),xyz),dot(vec3(.0557,-.2040,1.0570),xyz));rgb*=exposure;float lum=dot(rgb,vec3(.2126,.7152,.0722));rgb*=lum>1e-12?aces(lum)/max(lum,1e-12):0.;rgb=clamp(rgb,0.,1.);rgb=mix(12.92*rgb,1.055*pow(rgb,vec3(1./2.4))-.055,step(vec3(.0031308),rgb));frag=vec4(rgb,1);}`;
