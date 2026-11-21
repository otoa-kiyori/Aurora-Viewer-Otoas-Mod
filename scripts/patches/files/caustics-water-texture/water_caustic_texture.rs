//! Underwater caustics made from the environment's water (local patch set).
// aurora-local-patch: caustics-water-texture-module
//!
//! With "Use water texture" on, the refracted light map of the caustics is
//! built from the normal map the water surface draws (the EEP water's
//! texture, its two wave directions, its normal scale: env.wgsl water_pixel,
//! a port of the Legacy Viewer's waterF.glsl) instead of the fixed analytic
//! micro-waves of water_caustics.wgsl. Caustics are an Aurora effect: the
//! Legacy Viewer has none to follow.
//!
//! Two GPU passes per frame, no readback (water_caustics_texture.wgsl):
//! the surface's slope over one 25 m tile into a periodic slope map, then the
//! existing forward projection (flux-conserving area term, bounded fold)
//! with the lens Jacobian from finite differences of that map. The tile is
//! the big wave's own period; the little waves are rounded to whole repeats
//! per tile so it is seamless. The light map then holds 400 texels over
//! 25 m, the 16 per metre of the analytic one.
//!
//! The receivers (common.wgsl water_caustics) look the map up through the
//! surface's slow x warp, so the pattern stays under the ripples that make it
//! and neighbouring tiles are shifted against each other: no visible grid.
//! "Scale" divides the lookup position in both modes.
//!
//! Until the normal map is loaded (slot 0), or with the switch off, the
//! analytic pass, its 256² map and the receivers' original lines run
//! unchanged: off and scale 1 draw what upstream draws.

use super::{Uniform, WaterEffects, WaterGpu};
use bytemuck::{Pod, Zeroable};

/// Light-map texels of the analytic pattern (WaterGpu::new) and of the
/// texture's: 16 per metre both.
const ANALYTIC_MAP: u32 = 256;
const TEXTURE_MAP: u32 = 400;
/// The analytic tile (water_spectrum.wgsl: its modes are periodic over 16 m)
/// and the texture's: the big wave's period, 1 / 0.04 (env.wgsl).
const ANALYTIC_TILE: f32 = 16.0;
const TEXTURE_TILE: f32 = 25.0;
/// UV per metre of the surface's three layers (env.wgsl water_pixel: big,
/// little1, little2).
const LAYER_SCALES: [[f32; 2]; 3] = [[0.04, 0.04], [0.45, 0.9], [0.1, 0.2]];
/// Projection grid cells per side, by quality: the analytic pass draws
/// 96/160/224/256 over 18 m; over the texture's 28 m (25 + 2 x 1.5 margin)
/// these keep the same spacing (18.75 to 7 cm).
const TEXTURE_GRID: [u32; 4] = [150, 250, 350, 400];
/// The receivers' 4-tap filter, 0.4 texel (common.wgsl WATER_CAUSTIC_FILTER).
const FILTER_TEXELS: f32 = 0.4;
/// The range of the Scale slider: two octaves each way keep the light-map
/// texel between about 1.6 and 25 cm. Smaller repeats the analytic tile every
/// 4 m; larger blurs the pattern into soft blotches.
pub const SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.25..=4.0;

/// The caustic pattern scale a setting gives: within [`SCALE_RANGE`], 1 for
/// a value that is not a number (a hand-edited settings file).
pub fn caustic_scale(scale: f32) -> f32 {
    if scale.is_finite() {
        scale.clamp(*SCALE_RANGE.start(), *SCALE_RANGE.end())
    } else {
        1.0
    }
}

/// After the shared water uniform (offset 64): what the receivers and the
/// texture passes read (CausticControls in water_caustics_texture.wgsl, the
/// `caustic` field of WaterFxControls in common.wgsl).
#[repr(C)]
#[derive(Clone, Copy, PartialEq, Debug, Pod, Zeroable)]
pub(super) struct CausticUniform {
    /// pattern scale, tile (m), filter offset (uv), 1 = textured map
    lookup: [f32; 4],
    /// EEP wave1 direction xy, wave2 direction xy
    waves: [f32; 4],
    /// EEP normal scale xyz, w light-map area factor
    normal: [f32; 4],
    /// little1 repeats per tile xy, little2 xy
    repeats: [f32; 4],
    /// normal-map slot, grid cells per side, light-map size, 0
    source: [u32; 4],
}

impl CausticUniform {
    /// The analytic pattern at `scale`. At scale 1 its lookup is exactly the
    /// one the receivers' original lines use.
    fn analytic(scale: f32) -> Self {
        Self {
            lookup: [scale, ANALYTIC_TILE, FILTER_TEXELS / ANALYTIC_MAP as f32, 0.0],
            waves: [0.0; 4],
            normal: [0.0, 0.0, 0.0, area_factor(ANALYTIC_MAP, ANALYTIC_TILE)],
            repeats: [0.0; 4],
            source: [0, 0, ANALYTIC_MAP, 0],
        }
    }

    /// The pattern of the environment's normal map in `slot`.
    fn textured(scale: f32, water: &crate::WaterParams, slot: u32, quality: u8) -> Self {
        let r = repeats();
        let finite = |v: f32| if v.is_finite() { v } else { 0.0 };
        Self {
            lookup: [scale, TEXTURE_TILE, FILTER_TEXELS / TEXTURE_MAP as f32, 1.0],
            waves: [water.wave1.x, water.wave1.y, water.wave2.x, water.wave2.y].map(finite),
            normal: [
                finite(water.normal_scale.x),
                finite(water.normal_scale.y),
                finite(water.normal_scale.z),
                area_factor(TEXTURE_MAP, TEXTURE_TILE),
            ],
            repeats: [r[1][0], r[1][1], r[2][0], r[2][1]],
            source: [slot, TEXTURE_GRID[quality.min(3) as usize], TEXTURE_MAP, 0],
        }
    }

    fn textured_map(&self) -> bool {
        self.lookup[3] > 0.5
    }
}

/// The shared water uniform's contents at creation: zeros, then the analytic
/// pattern at scale 1, so that no frame ever reads a zero scale.
pub(super) fn initial_uniform() -> Vec<u8> {
    let mut bytes = bytemuck::bytes_of(&Uniform::zeroed()).to_vec();
    bytes.extend_from_slice(bytemuck::bytes_of(&CausticUniform::analytic(1.0)));
    bytes
}

/// (Texels per metre)²: a light-map pixel's projected source area times
/// this is 1 where the surface does not focus (fs_caustic's `area*256`).
fn area_factor(map: u32, tile: f32) -> f32 {
    let per_metre = map as f32 / tile;
    per_metre * per_metre
}

/// Whole repeats per texture tile of the surface's layers, nearest to their
/// real scale: the tile must be periodic for every layer at once.
fn repeats() -> [[f32; 2]; 3] {
    LAYER_SCALES.map(|s| s.map(|k| (k * TEXTURE_TILE).round().max(1.0)))
}

/// Whether the texture's pattern is drawn this frame: the switch is on, the
/// caustics are drawn (`flags`, which WaterGpu::prepare only keeps on valid
/// ground) and the environment's normal map is loaded. Otherwise the
/// analytic pattern stands in: never black caustics while the map loads.
fn textured(settings: WaterEffects, flags: u32, slot: u32) -> bool {
    settings.caustics_texture && flags & 32 != 0 && slot != 0
}

/// The texture passes' GPU objects, made the first time the switch is used.
struct Passes {
    slope_view: wgpu::TextureView,
    slope_pipeline: wgpu::RenderPipeline,
    slope_group: wgpu::BindGroup,
    project_pipeline: wgpu::RenderPipeline,
    project_group: wgpu::BindGroup,
}

/// WaterGpu's share of this patch (its `caustic` field). A WaterGpu made
/// again (water quality, region) starts from the analytic map.
#[derive(Default)]
pub(super) struct Caustics {
    passes: Option<Passes>,
    /// the scene's texture group of this frame, while the texture's pattern
    /// is drawn
    textures: Option<wgpu::BindGroup>,
    /// size of the light map in `caustics_view` (0: the 256² WaterGpu::new
    /// made)
    map: u32,
    written: Option<CausticUniform>,
    grid: u32,
}

/// A light map like WaterGpu::new's, `size` texels square.
fn light_map(device: &wgpu::Device, size: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("refracted water light"),
            size: wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&Default::default())
}

impl Passes {
    fn new(device: &wgpu::Device, uniform: &wgpu::Buffer, tex_layout: &wgpu::BindGroupLayout) -> Self {
        use crate::pipeline_cache::CachedPipelines;
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("water texture caustics"),
            source: wgpu::ShaderSource::Wgsl(crate::water_shader_source(
                &["water_caustics_texture.wgsl"],
                include_str!("shaders/water_caustics_texture.wgsl"),
            )),
        });
        let stages = wgpu::ShaderStages::VERTEX_FRAGMENT;
        let uniform_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let slope_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water texture caustics slope"),
            entries: &[uniform_entry],
        });
        let project_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("water texture caustics projection"),
            entries: &[
                uniform_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: stages,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: stages,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let slope = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("water texture slope"),
            size: wgpu::Extent3d {
                width: TEXTURE_MAP,
                height: TEXTURE_MAP,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rg16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let slope_view = slope.create_view(&Default::default());
        // periodic: the projection's margin reads the other side of the tile
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("water texture slope"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let slope_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water texture caustics slope"),
            layout: &slope_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform.as_entire_binding(),
            }],
        });
        let project_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("water texture caustics projection"),
            layout: &project_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&slope_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        let slope_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water texture caustics slope"),
            bind_group_layouts: &[Some(&slope_layout), Some(tex_layout)],
            immediate_size: 0,
        });
        let project_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("water texture caustics projection"),
            bind_group_layouts: &[Some(&project_layout)],
            immediate_size: 0,
        });
        let pipeline = |label, layout: &wgpu::PipelineLayout, vs, fs, format, blend| {
            device.create_render_pipeline_cached(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some(vs),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some(fs),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let additive = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::One,
            dst_factor: wgpu::BlendFactor::One,
            operation: wgpu::BlendOperation::Add,
        };
        Self {
            slope_pipeline: pipeline(
                "water texture slope",
                &slope_pipeline_layout,
                "vs_slope",
                "fs_slope",
                wgpu::TextureFormat::Rg16Float,
                None,
            ),
            project_pipeline: pipeline(
                "water texture caustics",
                &project_pipeline_layout,
                "vs_caustic_texture",
                "fs_caustic_texture",
                wgpu::TextureFormat::Rgba16Float,
                Some(wgpu::BlendState {
                    color: additive,
                    alpha: additive,
                }),
            ),
            slope_view,
            slope_group,
            project_group,
        }
    }
}

impl WaterGpu {
    /// After [`WaterGpu::prepare`]: which caustic pattern this frame draws,
    /// and its parameters for the receivers. `textures` is the scene's
    /// texture group (the normal map lives in its pages), `tex_layout` its
    /// layout. Returns whether the light map's view changed (the frame
    /// groups then rebind it), only when the pattern changes kind.
    pub(crate) fn prepare_caustics(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        settings: WaterEffects,
        water: &crate::WaterParams,
        textures: &wgpu::BindGroup,
        tex_layout: &wgpu::BindGroupLayout,
    ) -> bool {
        let scale = caustic_scale(settings.caustics_scale);
        let wanted = if textured(settings, self.flags, water.normal_texture) {
            CausticUniform::textured(scale, water, water.normal_texture, self.quality)
        } else {
            CausticUniform::analytic(scale)
        };
        let c = &mut self.caustic;
        let map = if wanted.textured_map() { TEXTURE_MAP } else { ANALYTIC_MAP };
        let resized = c.map.max(ANALYTIC_MAP) != map;
        if resized {
            self.caustics_view = light_map(device, map);
        }
        c.map = map;
        if wanted.textured_map() {
            if c.passes.is_none() {
                c.passes = Some(Passes::new(device, &self.uniform, tex_layout));
            }
            c.textures = Some(textures.clone());
        } else {
            c.textures = None;
        }
        c.grid = wanted.source[1];
        if c.written != Some(wanted) {
            queue.write_buffer(&self.uniform, std::mem::size_of::<Uniform>() as u64, bytemuck::bytes_of(&wanted));
            c.written = Some(wanted);
        }
        resized
    }

    /// The texture's pattern into the light map, when it is drawn this
    /// frame; false leaves the analytic pass to WaterGpu::encode.
    pub(super) fn encode_textured_caustics(&self, encoder: &mut wgpu::CommandEncoder) -> bool {
        let c = &self.caustic;
        let (Some(passes), Some(textures)) = (&c.passes, &c.textures) else {
            return false;
        };
        if self.flags & 32 == 0 {
            return false;
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("water texture slope"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &passes.slope_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&passes.slope_pipeline);
            pass.set_bind_group(0, &passes.slope_group, &[]);
            pass.set_bind_group(1, textures, &[]);
            pass.draw(0..3, 0..1);
        }
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("refracted water light (texture)"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.caustics_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&passes.project_pipeline);
        pass.set_bind_group(0, &passes.project_group, &[]);
        pass.draw(0..6, 0..c.grid * c.grid * 3);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_scale_stays_in_its_range_and_a_bad_value_is_one() {
        assert_eq!(caustic_scale(1.0), 1.0);
        assert_eq!(caustic_scale(2.5), 2.5);
        assert_eq!(caustic_scale(0.0), 0.25);
        assert_eq!(caustic_scale(-3.0), 0.25);
        assert_eq!(caustic_scale(100.0), 4.0);
        assert_eq!(caustic_scale(f32::NAN), 1.0);
        assert_eq!(caustic_scale(f32::INFINITY), 1.0);
    }

    #[test]
    fn the_analytic_pattern_at_scale_one_is_what_the_receivers_draw_upstream() {
        // common.wgsl takes its original lines for exactly these values:
        // fract(project / 16) and WATER_CAUSTIC_FILTER = 0.4 / 256
        let u = CausticUniform::analytic(1.0);
        assert_eq!(u.lookup, [1.0, 16.0, 0.4 / 256.0, 0.0]);
        assert!(!u.textured_map());
        // fs_caustic's area * 256
        assert_eq!(u.normal[3], 256.0);
        let bytes = initial_uniform();
        assert_eq!(bytes.len(), 64 + std::mem::size_of::<CausticUniform>());
        assert!(bytes[..64].iter().all(|b| *b == 0));
        assert_eq!(&bytes[64..], bytemuck::bytes_of(&u));
        // WGSL: five vec4 after the four of the shared uniform
        assert_eq!(std::mem::size_of::<CausticUniform>(), 80);
    }

    #[test]
    fn the_texture_tile_keeps_the_analytic_texel_and_flux_and_is_periodic_for_every_layer() {
        assert_eq!(area_factor(TEXTURE_MAP, TEXTURE_TILE), area_factor(ANALYTIC_MAP, ANALYTIC_TILE));
        let r = repeats();
        // the big wave repeats exactly once per tile
        assert_eq!(r[0], [1.0, 1.0]);
        assert_eq!(LAYER_SCALES[0][0] * TEXTURE_TILE, 1.0);
        for (layer, (repeats, scales)) in r.iter().zip(LAYER_SCALES).enumerate() {
            for (n, k) in repeats.iter().zip(scales) {
                assert_eq!(n.fract(), 0.0, "whole repeats: the tile is seamless");
                // a period within 20 % of the surface's own
                let period = TEXTURE_TILE / n;
                assert!((period * k - 1.0).abs() <= 0.2, "layer {layer}: {period} m against {} m", 1.0 / k);
            }
        }
        // the projection grid keeps the analytic spacing (18 m / 96..256)
        for (q, n) in TEXTURE_GRID.iter().enumerate() {
            let analytic = 18.0 / [96.0, 160.0, 224.0, 256.0][q];
            let texture = (TEXTURE_TILE + 3.0) / *n as f32;
            assert!((texture / analytic - 1.0).abs() < 0.05, "quality {q}");
        }
    }

    #[test]
    fn the_analytic_pattern_stands_in_until_the_normal_map_is_loaded() {
        let on = WaterEffects {
            caustics_texture: true,
            ..Default::default()
        };
        assert!(textured(on, 32, 7));
        assert!(!textured(on, 32, 0), "no normal map yet");
        assert!(!textured(on, 0, 7), "caustics not drawn (off, no ground)");
        assert!(!textured(WaterEffects::default(), 32, 7), "switch off by default");
        let water = crate::WaterParams {
            normal_scale: glam::Vec3::new(f32::NAN, 2.0, 2.0),
            ..Default::default()
        };
        let u = CausticUniform::textured(2.0, &water, 7, 9);
        assert!(u.textured_map());
        assert_eq!(u.lookup[..2], [2.0, 25.0]);
        assert_eq!(u.normal[..3], [0.0, 2.0, 2.0], "a NaN scale flattens, never poisons the map");
        assert_eq!(u.source, [7, 400, 400, 0], "quality clamped to ultra's grid");
    }

    #[test]
    fn the_texture_caustics_shader_is_valid() {
        let src = include_str!("shaders/water_caustics_texture.wgsl");
        let module = match naga::front::wgsl::parse_str(src) {
            Ok(m) => m,
            Err(e) => panic!("{}", e.emit_to_string(src)),
        };
        let mut v = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all());
        if let Err(e) = v.validate(&module) {
            panic!("{}", e.emit_to_string(src));
        }
    }
}
