// aurora-local-patch: shader-look-render-module
//! aurora-shader's effect chain as one layer of the photo's look stack:
//! the look "Otoa's Shader" (local patch set, 52-shader-photo-look).
//!
//! Its layer holds one operation, [`LookOp::UserChain`]. When `encode`
//! reaches it, the caller's [`UserChain`] draws the whole chain
//! (`user_post.rs`: every effect of `config.toml` in its order, with the
//! frame's depth, normals and the chain's own finish) from the image the
//! layers before left, and the layer's one pass lays that over it at the
//! layer's strength, like any look: the comparison line and the photo's
//! coverage come from the scene before the first layer, as for the others.
//!
//! The chain reads 8-bit gamma-encoded levels, what the layers leave one
//! another. The first layer's source is the frame's own input (the frame's
//! format, sRGB on some surfaces), so a chain asked first gets a plain
//! layer in front of it ([`padded`]) that writes it as the layers do: the
//! chain never reads the input itself.
//!
//! One chain a frame: it has one temporal history and one set of frame
//! uniforms, so a second "Otoa's Shader" layer leaves the image as it is.

use super::{LookLayer, LookOp, MAX_LAYERS, PhotoLook};
use std::sync::Arc;

/// Draws the chain at its layer's place, from that image (the chain's size,
/// 8-bit gamma-encoded); returns its result as the looks read it, or None
/// when the chain does not draw this frame (the layer then leaves the
/// image as it is).
pub(crate) type UserChain<'a> = dyn FnMut(&mut wgpu::CommandEncoder, &wgpu::TextureView) -> Option<wgpu::TextureView> + 'a;

/// The layer is "Otoa's Shader".
pub(crate) fn is_user_chain(layer: &LookLayer) -> bool {
    matches!(*layer.ops, [LookOp::UserChain])
}

/// The frame's looks hold "Otoa's Shader": the chain draws there, not at
/// its own place in the frame.
pub(crate) fn has_user_chain(look: &PhotoLook) -> bool {
    look.layers.iter().take(MAX_LAYERS).any(is_user_chain)
}

/// The layer draws the chain (at a strength of 0 the image stays as it is:
/// no need to draw it).
pub(super) fn draws(layer: &LookLayer) -> bool {
    is_user_chain(layer) && layer.intensity > 0.0
}

/// With the chain first, the layers behind a plain one that copies the
/// frame's input into the looks' own 8-bit images; None otherwise.
pub(super) fn padded(layers: &[LookLayer]) -> Option<Vec<LookLayer>> {
    if !layers.first().is_some_and(is_user_chain) {
        return None;
    }
    let plain = LookLayer {
        ops: Arc::from([LookOp::ColorTransform([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])]),
        intensity: 1.0,
        lut: None,
    };
    Some(std::iter::once(plain).chain(layers.iter().cloned()).take(MAX_LAYERS).collect())
}

#[cfg(test)]
mod tests {
    use super::super::{Looks, legacy_look};
    use super::*;
    use std::cell::Cell;

    fn layer(ops: Arc<[LookOp]>, intensity: f32) -> LookLayer {
        LookLayer { ops, intensity, lut: None }
    }

    fn chain_layer(intensity: f32) -> LookLayer {
        layer(Arc::from([LookOp::UserChain]), intensity)
    }

    fn look(layers: Vec<LookLayer>, size: (u32, u32)) -> PhotoLook {
        PhotoLook {
            layers,
            compare: None,
            image: [0.0, 0.0, size.0 as f32, size.1 as f32],
            scale: 1.0,
            hold: false,
            focus: None,
        }
    }

    #[test]
    fn the_chain_layer_is_known_and_never_first() {
        let negative = legacy_look("Negative").expect("Negative");
        let stack = look(vec![layer(negative.clone(), 1.0), chain_layer(0.5)], (4, 4));
        assert!(has_user_chain(&stack));
        assert!(!has_user_chain(&look(vec![layer(negative.clone(), 1.0)], (4, 4))));
        // not first: as it is
        assert!(padded(&stack.layers).is_none());
        // first: a plain layer in front, at full strength, the rest kept
        let first = [chain_layer(0.5), layer(negative.clone(), 1.0)];
        let padded = padded(&first).expect("padded");
        assert_eq!(padded.len(), 3);
        assert!(!is_user_chain(&padded[0]) && padded[0].intensity == 1.0);
        assert_eq!(padded[1..], first);
        // within the renderer's ceiling
        let full: Vec<LookLayer> = std::iter::once(chain_layer(1.0))
            .chain(std::iter::repeat_n(layer(negative, 1.0), MAX_LAYERS - 1))
            .collect();
        assert_eq!(super::padded(&full).map(|p| p.len()), Some(MAX_LAYERS));
        // a strength of 0 draws nothing
        assert!(draws(&chain_layer(0.1)) && !draws(&chain_layer(0.0)));
    }

    fn gpu() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::default();
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()
    }

    /// The looks drawn over `pixels` (Rgba8Unorm, rows going down), the
    /// chain being a solid `color` (its own image, cleared); the result,
    /// and how many times the chain drew.
    fn draw(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        look: &PhotoLook,
        size: (u32, u32),
        pixels: &[u8],
        color: Option<wgpu::Color>,
    ) -> (Vec<u8>, u32) {
        let format = wgpu::TextureFormat::Rgba8Unorm;
        let mut looks = Looks::new(device);
        looks.prepare(device, format, size);
        let input = looks.targets.as_ref().map(|t| t.input.texture().clone()).expect("prepared");
        let extent = wgpu::Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        };
        let row = size.0 * 4;
        let layout = wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(row),
            rows_per_image: Some(size.1),
        };
        queue.write_texture(input.as_image_copy(), pixels, layout, extent);
        let padded = row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let texture = |usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some("user chain test"),
                size: extent,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let target = texture(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let solid = texture(wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING).create_view(&Default::default());
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("user chain test"),
            size: crate::readback::readback_size((padded * size.1) as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let drawn = Cell::new(0u32);
        let mut chain = |encoder: &mut wgpu::CommandEncoder, _input: &wgpu::TextureView| {
            let color = color?;
            drawn.set(drawn.get() + 1);
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("user chain test"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &solid,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            Some(solid.clone())
        };
        let mut encoder = device.create_command_encoder(&Default::default());
        let view = target.create_view(&Default::default());
        looks.encode_chained(device, queue, &mut encoder, look, &view, None, Some(&mut chain));
        encoder.copy_texture_to_buffer(
            target.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &read,
                layout: wgpu::TexelCopyBufferLayout {
                    bytes_per_row: Some(padded),
                    ..layout
                },
            },
            extent,
        );
        queue.submit([encoder.finish()]);
        read.map_async(wgpu::MapMode::Read, .., |_| {});
        device.poll(wgpu::PollType::wait_indefinitely()).expect("poll");
        let mapped = read.get_mapped_range(..).expect("mapped");
        let out = mapped
            .chunks_exact(padded as usize)
            .flat_map(|r| &r[..row as usize])
            .copied()
            .collect();
        (out, drawn.get())
    }

    fn image(w: u32, h: u32) -> Vec<u8> {
        (0..w * h)
            .flat_map(|i| {
                let (x, y) = (i % w, i / w);
                [(x * 255 / w) as u8, (y * 255 / h) as u8, 96, 255]
            })
            .collect()
    }

    /// The pixels' colours, all alike.
    fn every(pixels: &[u8], rgb: [u8; 3]) -> bool {
        pixels.as_chunks::<4>().0.iter().all(|p| p[..3] == rgb)
    }

    #[test]
    fn the_chain_draws_at_its_layer_s_place_and_strength() {
        let _gpu = crate::test_gpu::lock();
        let Some((device, queue)) = gpu() else {
            return;
        };
        let (w, h) = (32u32, 16u32);
        let input = image(w, h);
        let red = Some(wgpu::Color::RED);
        let negative = legacy_look("Negative").expect("Negative");
        // after the negative: the chain's red over it
        let after = look(vec![layer(negative.clone(), 1.0), chain_layer(1.0)], (w, h));
        let (got, drawn) = draw(&device, &queue, &after, (w, h), &input, red);
        assert_eq!(drawn, 1);
        assert!(every(&got, [255, 0, 0]), "{:?}", &got[..8]);
        // before it: the negative of red
        let before = look(vec![chain_layer(1.0), layer(negative.clone(), 1.0)], (w, h));
        let (got, _) = draw(&device, &queue, &before, (w, h), &input, red);
        assert!(every(&got, [0, 255, 255]), "{:?}", &got[..8]);
        // alone at 0: the image as it is, the chain not drawn
        let (got, drawn) = draw(&device, &queue, &look(vec![chain_layer(0.0)], (w, h)), (w, h), &input, red);
        assert_eq!((got.as_slice(), drawn), (input.as_slice(), 0));
        // at half: halfway between the image and the chain's result
        let (got, _) = draw(&device, &queue, &look(vec![chain_layer(0.5)], (w, h)), (w, h), &input, red);
        for (a, b) in input.as_chunks::<4>().0.iter().zip(got.as_chunks::<4>().0) {
            for (k, c) in [255.0f32, 0.0, 0.0].into_iter().enumerate() {
                assert!((f32::from(a[k]) * 0.5 + c * 0.5 - f32::from(b[k])).abs() <= 1.0, "{a:?} {b:?}");
            }
        }
        // the chain off this frame (the master switch): as it is
        let (got, _) = draw(&device, &queue, &look(vec![chain_layer(1.0)], (w, h)), (w, h), &input, None);
        assert_eq!(got, input);
        // two of them: one chain a frame, the second leaves the image
        let twice = look(vec![chain_layer(1.0), layer(negative, 1.0), chain_layer(1.0)], (w, h));
        let (got, drawn) = draw(&device, &queue, &twice, (w, h), &input, red);
        assert_eq!(drawn, 1);
        assert!(every(&got, [0, 255, 255]), "{:?}", &got[..8]);
        // the comparison line: the scene as it is left of it
        let mut compared = look(vec![chain_layer(1.0)], (w, h));
        compared.compare = Some(0.25);
        let (got, _) = draw(&device, &queue, &compared, (w, h), &input, red);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let at = (y * w as usize + x) * 4;
                let want: &[u8] = if x < w as usize / 4 { &input[at..at + 3] } else { &[255, 0, 0] };
                assert_eq!(&got[at..at + 3], want, "{x}, {y}");
            }
        }
    }
}
