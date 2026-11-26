enable wgpu_binding_array;
// aurora-local-patch: caustics-water-texture-shader
// Caustics made from the environment's own water (local patch set,
// water_caustic_texture.rs). The same forward projection as
// water_caustics.wgsl, with its analytic micro-waves replaced by the normal
// map the surface draws (env.wgsl water_pixel, a port of waterF.glsl): its
// three layers, their scroll and directions, the normal scale.
//
// 1. vs_slope / fs_slope: the surface's slope over one tile of the warped
//    texture space the surface samples (its x warp is applied by the
//    receivers), into a periodic slope map. A fragment shader, because the
//    bindless pages are only visible to fragments; the mip comes from the
//    derivatives, like the surface's.
// 2. vs_caustic_texture / fs_caustic_texture: refracted triangles of that
//    slope field add light, conserving their source area; the lens Jacobian
//    comes from central differences of the slope map.

struct CausticControls {
    area:vec4<f32>, previous:vec4<f32>, clock:vec4<f32>, counts:vec4<u32>,
    // receivers: x pattern scale, y tile (m), z filter offset (uv), w textured
    lookup:vec4<f32>,
    // wave1 direction xy, wave2 direction xy (EEP)
    waves:vec4<f32>,
    // EEP normal scale xyz, w light-map area factor ((texels per m)^2)
    normal:vec4<f32>,
    // repeats per tile of the little waves: little1 xy, little2 xy (the big
    // wave repeats once)
    repeats:vec4<f32>,
    // x normal-map slot, y grid cells per side, z light-map size, w the
    // receivers' dimming (an f32's bits, not read here)
    source:vec4<u32>,
};
@group(0) @binding(0) var<uniform> caustic_fx:CausticControls;
@group(0) @binding(1) var slope_map:texture_2d<f32>;
@group(0) @binding(2) var slope_sampler:sampler;
// While the environment cross-fades two water normal maps (patch 998), the
// slope pass's second map (water_caustic_texture.rs CausticFade); zero
// otherwise, its taps then not taken.
struct CausticFade {
    // the first map's own normal scale xyz, w 1 while the water cross-fades
    first:vec4<f32>,
    // the second map's waves: wave1 direction xy, wave2 direction xy
    waves:vec4<f32>,
    // its normal scale xyz, w its share (0: the first map alone)
    next:vec4<f32>,
    // x its normal-map slot
    source:vec4<u32>,
};
@group(0) @binding(3) var<uniform> caustic_fade:CausticFade;
@group(1) @binding(0) var tex_pages:binding_array<texture_2d_array<f32>>;
@group(1) @binding(1) var tex_sampler:sampler;
// the texture locations start with the bound page count (common.wgsl
// TextureLocations, upstream #2657); a page past it reads reserved page 0,
// as common.wgsl texture_page_index does
struct TextureLocations {
    page_count:u32,
    values:array<u32>,
};
@group(1) @binding(2) var<storage,read> tex_loc:TextureLocations;

fn texture_page_index(loc:u32)->u32 {
    let page=loc>>16u;
    let last_bound_page=max(tex_loc.page_count,1u)-1u;
    return select(page,0u,page>last_bound_page);
}

// the projection's source extends this far past the tile on each side, so
// that the light a ray carries across an edge comes back from the other one
const CAUSTIC_MARGIN:f32=1.5;
// steepest slope kept: at the deepest layer (7 m) a ray then moves at most
// 7 x 0.25 x 0.8 = 1.4 m, within the margin
const CAUSTIC_MAX_SLOPE:f32=0.8;

@vertex
fn vs_slope(@builtin(vertex_index) vertex:u32)->@builtin(position) vec4<f32> {
    let uv=vec2<f32>(f32((vertex<<1u)&2u),f32(vertex&2u));
    return vec4<f32>(uv*2.0-1.0,0.0,1.0);
}

fn water_layer_at(slot:u32,uv:vec2<f32>)->vec3<f32> {
    let loc=tex_loc.values[slot];
    return textureSample(tex_pages[texture_page_index(loc)],tex_sampler,uv,loc&0xffffu).xyz*2.0-vec3<f32>(1.0);
}

fn water_layer(uv:vec2<f32>)->vec3<f32> {
    return water_layer_at(caustic_fx.source.x,uv);
}

@fragment
fn fs_slope(@builtin(position) pixel:vec4<f32>)->@location(0) vec4<f32> {
    let tile=caustic_fx.lookup.y;
    // metres of the warped texture space (env.wgsl's `v`)
    let v=pixel.xy/f32(caustic_fx.source.z)*tile;
    let t=caustic_fx.clock.x*0.5;
    let w1d=caustic_fx.waves.xy;let w2d=caustic_fx.waves.zw;
    // the surface's layers, their scale rounded to whole repeats per tile
    let big=water_layer(v/tile+w1d*t*0.055);
    let little1=water_layer(v*caustic_fx.repeats.xy/tile+w2d*t*0.13);
    let little2=water_layer(v*caustic_fx.repeats.zw/tile+w1d*t*0.1);
    let wavef=(big+little1*0.4+little2*0.6)*0.5;
    let scale=caustic_fx.normal.xyz;
    // height gradient of the shaded surface: with the enhanced waves it is
    // (n.xy * 0.18, 1); without, the normal scaled and its z doubled
    var gradient=-0.18*scale.xy*wavef.xy;
    if((caustic_fx.counts.z&1u)==0u){
        let n=wavef*scale;
        gradient=-n.xy/max(n.z*2.0,0.05);
    }
    // While the water cross-fades (patch 998) the first map keeps its own
    // normal scale and the second map's layers, at its own waves and scale,
    // are mixed in by its share, as the surface mixes them (env.wgsl
    // water_pixel); the gradient is then taken from the mix.
    if(caustic_fade.first.w>0.5){
        var n=wavef*caustic_fade.first.xyz;
        let share=caustic_fade.next.w;
        if(share>0.0){
            let slot=caustic_fade.source.x;
            let n1d=caustic_fade.waves.xy;let n2d=caustic_fade.waves.zw;
            let b0=water_layer_at(slot,v/tile+n1d*t*0.055);
            let b1=water_layer_at(slot,v*caustic_fx.repeats.xy/tile+n2d*t*0.13);
            let b2=water_layer_at(slot,v*caustic_fx.repeats.zw/tile+n1d*t*0.1);
            n=mix(n,(b0+b1*0.4+b2*0.6)*0.5*caustic_fade.next.xyz,share);
        }
        gradient=-0.18*n.xy;
        if((caustic_fx.counts.z&1u)==0u){
            gradient=-n.xy/max(n.z*2.0,0.05);
        }
    }
    gradient*=min(1.0,CAUSTIC_MAX_SLOPE/max(length(gradient),1e-6));
    return vec4<f32>(gradient,0.0,0.0);
}

fn slope_at(p:vec2<f32>)->vec2<f32> {
    return textureSampleLevel(slope_map,slope_sampler,p/caustic_fx.lookup.y,0.0).xy;
}

struct CausticOut {
    @builtin(position) clip:vec4<f32>,
    @location(0) source:vec2<f32>,
    @location(1) @interpolate(flat) layer:u32,
};

@vertex
fn vs_caustic_texture(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->CausticOut {
    let n=caustic_fx.source.y;let count=n*n;
    let layer=instance/count;let cell=instance%count;
    let corners=array<vec2<f32>,6>(vec2<f32>(0.0,0.0),vec2<f32>(1.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,0.0),vec2<f32>(1.0,1.0),vec2<f32>(0.0,1.0));
    let tile=caustic_fx.lookup.y;
    let p=(vec2<f32>(f32(cell%n),f32(cell/n))+corners[vertex])/f32(n)*(tile+2.0*CAUSTIC_MARGIN)-CAUSTIC_MARGIN;
    let normal=normalize(vec3<f32>(-slope_at(p),1.0));
    let ray=refract(vec3<f32>(0.0,0.0,-1.0),normal,0.7501875);
    let depths=array<f32,3>(0.75,2.5,7.0);
    let hit=(p+ray.xy*depths[layer]/max(-ray.z,0.2))*(2.0/tile);
    var out:CausticOut;out.clip=vec4<f32>(hit.x-1.0,1.0-hit.y,0.0,1.0);
    out.source=p;out.layer=layer;return out;
}

@fragment
fn fs_caustic_texture(in:CausticOut)->@location(0) vec4<f32> {
    let dx=dpdx(in.source);let dy=dpdy(in.source);
    let area=abs(dx.x*dy.y-dx.y*dy.x);
    let depths=array<f32,3>(0.75,2.5,7.0);
    // the Hessian of the surface from the slope map, one texel each way
    let step=caustic_fx.lookup.y/f32(caustic_fx.source.z);
    let x0=slope_at(in.source-vec2<f32>(step,0.0));let x1=slope_at(in.source+vec2<f32>(step,0.0));
    let y0=slope_at(in.source-vec2<f32>(0.0,step));let y1=slope_at(in.source+vec2<f32>(0.0,step));
    let hessian=vec3<f32>(x1.x-x0.x,((y1.x-y0.x)+(x1.y-x0.y))*0.5,y1.y-y0.y)/(2.0*step);
    let curvature=hessian*depths[in.layer]*0.25;
    let jacobian=(1.0+curvature.x)*(1.0+curvature.z)-curvature.y*curvature.y;
    // as water_caustics.wgsl: projected area anchors the flux, the bounded
    // fold stands for the sun's disc and the pixel footprint
    let focus=inverseSqrt(jacobian*jacobian+0.015);
    let intensity=min(mix(area*caustic_fx.normal.w,focus,0.85),6.0);
    return vec4<f32>(select(0.0,intensity,in.layer==0u),select(0.0,intensity,in.layer==1u),select(0.0,intensity,in.layer==2u),0.0);
}
