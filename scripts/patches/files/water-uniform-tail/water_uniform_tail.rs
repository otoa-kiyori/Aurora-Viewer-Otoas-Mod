// aurora-local-patch: water-uniform-tail-module
//! The shared water uniform's local tail (local patch set).
//!
//! The water effects' uniform (binding 17, `water_fx` in common.wgsl) holds
//! upstream's own 64 bytes (`Uniform`), then one fixed region per local
//! water feature, whether its patch is there or not, so that each feature
//! writes only its own region and none needs another for its place:
//! - the caustics' five vec4 (water_caustic_texture.rs): `caustic`, then
//!   `caustic_rest`;
//! - the open-sea swell's five (ocean_swell.rs): `ocean`, `ocean_area`,
//!   `ocean_shape`, `ocean_fine`, `ocean_next`;
//! - the water shape's three (water_shape.rs): `shape`, `shape_next`,
//!   `shape_fade`.
//!
//! The whole buffer starts zero, which every region reads as off; a region
//! whose feature is absent stays zero and nothing reads it. Cost: 208 bytes
//! more in one uniform buffer, nothing per frame.

/// Where the caustics' region starts: after upstream's own `Uniform`.
pub(super) const CAUSTIC_OFFSET: u64 = 64;
/// Its size (bytes): five vec4.
pub(super) const CAUSTIC_BYTES: u64 = 80;
/// Where the open-sea swell's region starts.
pub(super) const OCEAN_OFFSET: u64 = CAUSTIC_OFFSET + CAUSTIC_BYTES;
/// Its size (bytes): five vec4.
pub(super) const OCEAN_BYTES: u64 = 80;
/// Where the water shape's region starts.
pub(super) const SHAPE_OFFSET: u64 = OCEAN_OFFSET + OCEAN_BYTES;
/// Its size (bytes): three vec4.
pub(super) const SHAPE_BYTES: u64 = 48;
/// The whole uniform (bytes).
pub(super) const SIZE: u64 = SHAPE_OFFSET + SHAPE_BYTES;

/// The shared water uniform's contents at creation: zeros (every region
/// off).
pub(super) fn initial_uniform() -> Vec<u8> {
    vec![0; SIZE as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_regions_follow_upstreams_uniform_in_the_shaders_order() {
        assert_eq!(CAUSTIC_OFFSET as usize, std::mem::size_of::<super::super::Uniform>());
        let bytes = initial_uniform();
        assert_eq!(bytes.len() as u64, SIZE);
        assert!(bytes.iter().all(|b| *b == 0));
        // common.wgsl's WaterFxControls, as naga lays it out: each region's
        // first field at its offset, the struct as large as the buffer
        let src = format!(
            "{}\n{}\n{}",
            crate::COMMON_SHADER,
            include_str!("shaders/object.wgsl"),
            include_str!("shaders/env.wgsl")
        );
        let module = naga::front::wgsl::parse_str(&src).expect("the scene shader parses");
        let controls = module
            .types
            .iter()
            .find(|(_, t)| t.name.as_deref() == Some("WaterFxControls"))
            .expect("WaterFxControls");
        let naga::TypeInner::Struct { members, span } = &controls.1.inner else {
            panic!("WaterFxControls is a struct");
        };
        assert_eq!(u64::from(*span), SIZE);
        let offset = |name: &str| {
            members
                .iter()
                .find(|m| m.name.as_deref() == Some(name))
                .map(|m| u64::from(m.offset))
        };
        assert_eq!(offset("caustic"), Some(CAUSTIC_OFFSET));
        assert_eq!(offset("caustic_rest"), Some(CAUSTIC_OFFSET + 16));
        assert_eq!(offset("ocean"), Some(OCEAN_OFFSET));
        assert_eq!(offset("ocean_next"), Some(OCEAN_OFFSET + OCEAN_BYTES - 16));
        assert_eq!(offset("shape"), Some(SHAPE_OFFSET));
        assert_eq!(offset("shape_fade"), Some(SIZE - 16));
    }
}
