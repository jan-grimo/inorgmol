/// Modeling for embedding a single shape
pub mod solitary_shape {

use crate::strong::{IndexBase, Index};
use itertools::Itertools;
use crate::dg::DistanceBounds;
use crate::dg::refinement::Chiral;
use crate::shapes::{Shape, Vertex, Particle};

fn particle_index(p: &Particle, shape: &Shape) -> usize {
    match p {
        Particle::Vertex(v) => v.get(),
        Particle::Origin => shape.num_vertices()
    }
}

/// Generate distance bounds from a shape
pub fn shape_into_bounds(shape: &Shape) -> DistanceBounds {
    // Include the origin particle
    let mut bounds = DistanceBounds::new(shape.num_vertices() + 1);

    const TOLERANCE_PERCENT: f64 = 1.0;

    let mut particles: Vec<Particle> = Vertex::range(shape.num_vertices()).map(Particle::Vertex).collect();
    particles.push(Particle::Origin);

    // Distances between all particles
    for (a, b) in particles.iter().tuple_combinations() {
        let distance = (shape.particle_position(*a) - shape.particle_position(*b)).norm();
        let delta = distance * TOLERANCE_PERCENT / 100.0;

        let [i, j] = [a, b].map(|p| particle_index(p, shape));

        bounds.increase_lower_bound(i, j, distance - delta);
        bounds.decrease_upper_bound(i, j, distance + delta);
    }

    bounds.floyd_triangle_smooth().expect("Valid bounds")
}

/// Generate a chiral constraint from a shape tetrahedron
pub fn chiral_from_tetrahedron(tetrahedron: [Particle; 4], shape: &Shape, relative_tolerance: f64) -> Chiral<f64> {
    let adjusted_volume = 6.0 * crate::geometry::signed_tetrahedron_volume_with_array(tetrahedron.map(|p| shape.particle_position(p)));
    debug_assert!(adjusted_volume > 0.0);
    let sites = tetrahedron.map(|p| vec![particle_index(&p, shape)]);
    let weight: f64 = 1.0;
    let tolerance = relative_tolerance * adjusted_volume;
    let adjusted_volume_bounds = (adjusted_volume - tolerance, adjusted_volume + tolerance);
    Chiral {sites, adjusted_volume_bounds, weight}
}

}
