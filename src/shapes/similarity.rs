// TODO
// - Consider trying to homogenize all the Case implementations
// - Add (automatic?) centroid prematching

extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use thiserror::Error;
use memoize::memoize;
use itertools::Itertools;
use std::collections::HashMap;

use crate::strong::matrix::StrongPoints;
use crate::strong::bijection::{Bijection, bijections};
use crate::permutation::{Permutation, permutations, slice_next};
use crate::shapes::*;

use std::convert::TryFrom;

pub fn unit_sphere_normalize(mut x: Matrix3N) -> Matrix3N {
    // Remove centroid
    let centroid = x.column_mean();
    for mut v in x.column_iter_mut() {
        v -= centroid;
    }

    // Rescale all distances so that the longest is a unit vector
    let max_norm: f64 = x.column_iter().map(|v| v.norm()).fold(0.0, |a, b| a.max(b));
    for mut v in x.column_iter_mut() {
        v /= max_norm;
    }

    x
}

pub fn apply_permutation_static<C: nalgebra::base::Dim, S>(x: &na::Matrix<f64, na::Const<3>, C, S>, p: &Permutation) -> na::Matrix<f64, na::Const<3>, C, <na::DefaultAllocator as na::allocator::Allocator<f64, na::Const<3>, C>>::Buffer>
where 
    S: na::RawStorage<f64, na::Const<3>, C>,
    C: na::DimName,
    na::DefaultAllocator: na::allocator::Allocator<f64, na::Const<3>, C>
{

    assert_eq!(x.ncols(), p.set_size());
    let inverse = p.inverse();
    na::OMatrix::<f64, na::Const<3>, C>::from_fn(|i, j| x[(i, inverse[j])])
}

// Column-permute a matrix
//
// Post-condition is x.column(i) == result.column(p(i))
pub fn apply_permutation(x: &Matrix3N, p: &Permutation) -> Matrix3N {
    assert_eq!(x.ncols(), p.set_size());
    let inverse = p.inverse();
    Matrix3N::from_fn(x.ncols(), |i, j| x[(i, inverse[j])])
}

mod scaling {
    use crate::shapes::Matrix3N;

    use argmin::core::{CostFunction, Error, Executor};
    use argmin::solver::brent::BrentOpt;

    fn msd(cloud: &Matrix3N, shape: &Matrix3N, factor: f64) -> f64 {
        (shape.scale(factor) - cloud)
            .column_iter()
            .map(|col| col.norm_squared())
            .sum()
    }

    struct CoordinateScalingProblem<'a> {
        pub cloud: &'a Matrix3N,
        pub shape: &'a Matrix3N
    }

    impl CostFunction for CoordinateScalingProblem<'_> {
        type Param = f64;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(msd(self.cloud, self.shape, *p))
        }
    }

    fn csm_from_msd(cloud: &Matrix3N, msd: f64) -> f64 {
        100.0 * msd / cloud.column_iter().map(|v| v.norm_squared()).sum::<f64>()
    }

    pub struct Minimum {
        pub factor: f64,
        pub msd: f64
    }

    pub fn minimize(cloud: &Matrix3N, shape: &Matrix3N) -> Minimum {
        const DEFAULT_MIN: f64 = 0.3;
        const DEFAULT_MAX: f64 = 1.8;
        let precondition = direct_factor(cloud, shape);
        assert!((DEFAULT_MIN..=DEFAULT_MAX).contains(&precondition));

        let phi_sectioning = BrentOpt::new(DEFAULT_MIN, DEFAULT_MAX);
        let scaling_problem = CoordinateScalingProblem { cloud, shape };
        let result = Executor::new(scaling_problem, phi_sectioning)
            .configure(|state| state.param(precondition).max_iters(100))
            .run()
            .unwrap();

        Minimum {factor: result.state.best_param.unwrap(), msd: result.state.best_cost}
    }

    pub fn minimize_csm(cloud: &Matrix3N, shape: &Matrix3N) -> f64 {
        csm_from_msd(cloud, minimize(cloud, shape).msd)
    }

    // Final scale formula of section 2E in "Closed-form solution of absolute orientation with
    // quaternions" by Berthold K.P. Horn in J. Opt. Soc. Am. A, 1986
    //
    // Works (empirically) only for undistorted cases, but useful for 
    // preconditioning the minimization
    pub fn direct_factor(cloud: &Matrix3N, shape: &Matrix3N) -> f64 {
        (cloud.column_iter().map(|v| v.norm_squared()).sum::<f64>()
            / shape.column_iter().map(|v| v.norm_squared()).sum::<f64>()).sqrt()
    }
}

// Bool matrix indicating which vertex pairs are rotationally unique
#[memoize]
pub fn skip_vertices(shape_name: Name) -> na::DMatrix<bool> {
    let shape = shape_from_name(shape_name);

    let s = shape.size();
    let mut skips = na::DMatrix::<bool>::from_element(s + 1, s + 1, true);
    let rotations = shape.generate_rotations();
    let viable_vertices: Vec<Vertex> = shape.vertex_groups()
        .iter()
        .map(|g| *g.first().unwrap())
        .collect();
    for i in viable_vertices {
        let vertex_groups = shape.vertex_groups_holding(&[i], &rotations);
        for group in vertex_groups {
            let j = group.first().expect("Vertex groups shouldn't be empty");
            skips[(i.get(), j.get())] = false;
        }

        // Centroid is always a valid second match
        skips[(i.get(), s)] = false;
    }

    // If matching centroid first, reuse viable_vertices for second match
    for i in 0..s {
        skips[(s, i)] = skips[(i, 0)];
    }

    skips
}

struct SkipsBijectionGenerator {
    starting_pairs: Vec<(Vertex, Vertex)>,
    maybe_next: Option<Bijection<Column, Vertex>>
}

impl SkipsBijectionGenerator {
    pub fn new(shape_name: Name) -> SkipsBijectionGenerator {
        let skips = skip_vertices(shape_name);
        let s = skips.ncols();

        let mut pairs: Vec<(Vertex, Vertex)> = (0..s).cartesian_product(0..s)
            .filter(|(i, j)| !skips[(*i, *j)] && i != j)
            .map(|(i, j)| (Vertex(i), Vertex(j)))
            .collect();

        pairs.reverse();

        let mut generator = SkipsBijectionGenerator {
            starting_pairs: pairs,
            maybe_next: Some(Bijection::identity(s))
        };
        generator.reset_from_next_pair();
        generator
    }

    pub fn reset_from_next_pair(&mut self) {
        if let Some((a, b)) = self.starting_pairs.pop() {
            let mut initial = Vec::new();
            initial.push(a.get());
            initial.push(b.get());

            let previous = self.maybe_next.as_ref();
            let s = previous.expect("As long as there's pairs, this should be set").set_size();

            for i in 0..s {
                if i != a.get() && i != b.get() {
                    initial.push(i);
                }
            }

            let first = Permutation::try_from(initial).unwrap();

            self.maybe_next = Some(Bijection::new(first));
        } else {
            self.maybe_next = None;
        }
    }
}

impl Iterator for SkipsBijectionGenerator {
    type Item = Bijection<Column, Vertex>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.maybe_next.is_none() {
            return None;
        }

        let cached_maybe_next = self.maybe_next.clone();
        if !slice_next(&mut self.maybe_next.as_mut().unwrap().permutation.sigma[2..]) {
            self.reset_from_next_pair();
        }

        cached_maybe_next
    } 
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SimilarityError {
    #[error("Number of particles does not match shape size")]
    ParticleNumberMismatch,
}

pub struct Similarity {
    pub bijection: Bijection<Vertex, Column>,
    pub csm: f64,
}

/// Polyhedron similarity performing quaternion fits on all rotationally distinct bijections
pub fn polyhedron_reference_base<const USE_SKIPS: bool>(x: Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::ParticleNumberMismatch);
    }

    let cloud = StrongPoints::new(unit_sphere_normalize(x));

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = StrongPoints::new(unit_sphere_normalize(shape_coordinates));

    type BijectionGenerator = dyn Iterator<Item = Bijection<Column, Vertex>>;
    let bijection_generator: Box<BijectionGenerator> = match USE_SKIPS {
        true => Box::new(SkipsBijectionGenerator::new(s)),
        false => Box::new(bijections(n))
    };

    let (best_bijection, best_fit) = bijection_generator
        .map(|p| {
            let fit = shape_coordinates.quaternion_fit_with_rotor(&cloud.apply_bijection(&p));
            (p, fit)
        })
        .min_by(|(_, fit_a), (_, fit_b)| fit_a.msd.partial_cmp(&fit_b.msd).expect("No NaNs in MSDs"))
        .expect("There is always at least one permutation available to try");

    let best_bijection = best_bijection.inverse();

    let permuted_shape = shape_coordinates.apply_bijection(&best_bijection);
    let rotated_shape = best_fit.rotate_stator(&permuted_shape.matrix);

    let csm = scaling::minimize_csm(&cloud.matrix, &rotated_shape);
    Ok(Similarity {bijection: best_bijection, csm})
}

pub fn polyhedron_reference(x: Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    polyhedron_reference_base::<true>(x, s)
}

pub mod linear_assignment {
    use std::convert::TryFrom;
    use crate::shapes::similarity::{Permutation, permutations};

    /// Brute force the linear assignment problem (factorial complexity)
    pub fn brute_force(v: usize, cost_fn: &dyn Fn(usize, usize) -> f64) -> Permutation {
        let costs = nalgebra::DMatrix::from_fn(v, v, cost_fn);
        let (_, permutation) = permutations(v)
            .map(|p| ((0..v).map(|i| costs[(i, p[i])]).sum(), p) )
            .min_by(|(a, _): &(f64, Permutation), (b, _): &(f64, Permutation)| a.partial_cmp(b).expect("Encountered NaNs"))
            .expect("Always at least one permutation available to try");

        permutation
    }

    /// Use Jonker-Volgenant solution (cubic complexity)
    pub fn jonker_volgenant(v: usize, cost_fn: &dyn Fn(usize, usize) -> f64) -> Permutation {
        let costs_ndarray = lapjv::Matrix::from_shape_fn((v, v), |(i, j)| cost_fn(i, j));
        let (forward_sigma, _) = lapjv::lapjv(&costs_ndarray).expect("lapjv doesn't fail");
        Permutation::try_from(forward_sigma).expect("lapjv yields valid permutations")
    }

    /// Switch depending on matrix size
    pub fn optimal(v: usize, cost_fn: &dyn Fn(usize, usize) -> f64) -> Permutation {
        if v > 3 {
            jonker_volgenant(v, cost_fn)
        } else {
            brute_force(v, cost_fn)
        }
    }

    #[cfg(test)]
    mod tests {
        use crate::shapes::similarity::linear_assignment::*;
        
        #[test]
        fn jv_yields_forward_permutations() {
            let mat = nalgebra::DMatrix::from_row_slice(5, 5, &[
                1.8532, 0.7544, 0.3690, 2.3446, 3.4903, 
                1.7034, 0.1503, 2.2695, 0.6570, 2.8954, 
                3.4626, 3.5621, 3.3862, 1.3924, 0.0061, 
                0.0682, 1.0255, 1.6610, 2.5187, 3.3430, 
                0.9999, 0.9999, 0.9999, 1.0000, 1.0000 
            ]);
            let cost_fn = |i, j| mat[(i, j)];

            assert_eq!(brute_force(5, &cost_fn), jonker_volgenant(5, &cost_fn));
        }
    }
}

/// Polyhedron similarity performing quaternion fits only on a limited number of 
/// vertices before assigning the rest by brute force linear assignment
pub fn polyhedron_base<const PREMATCH: usize, const USE_SKIPS: bool, const LAP_JV: bool>(x: Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    const MIN_PREMATCH: usize = 2;
    assert!(MIN_PREMATCH <= PREMATCH); // TODO trigger compilation failure? static assert?

    type Occupation = Bijection<Vertex, Column>;
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::ParticleNumberMismatch);
    }

    if n <= PREMATCH {
        return polyhedron_reference_base::<USE_SKIPS>(x, s);
    }

    let cloud = StrongPoints::<Column>::new(unit_sphere_normalize(x));
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);
    let shape_coordinates = StrongPoints::<Vertex>::new(shape_coordinates);

    type PartialPermutation = HashMap<Column, Vertex>;

    struct PartialMsd {
        msd: f64,
        mapping: PartialPermutation,
    }
    let left_free: Vec<Column> = (PREMATCH..n).map(Column::from).collect();

    let skips = USE_SKIPS.then(|| skip_vertices(s));
    let narrow = (0..n)
        .map_into::<Vertex>()
        .permutations(PREMATCH)
        .filter(|vertices| {
            match USE_SKIPS {
                true => !skips.as_ref().unwrap()[(vertices[0].get(), vertices[1].get())],
                false => true
            }
        })
        .fold(
            PartialMsd {msd: f64::MAX, mapping: PartialPermutation::new()},
            |best, vertices| -> PartialMsd {
                let mut partial_map = PartialPermutation::with_capacity(n);
                vertices.iter().enumerate().for_each(|(c, v)| { partial_map.insert(Column::from(c), *v); });
                let partial_fit = cloud.quaternion_fit_with_map(&shape_coordinates, &partial_map);

                // If the msd caused only by the partial map is already worse, skip
                if partial_fit.msd > best.msd {
                    return best;
                }

                // Assemble cost matrix of matching each pair of unmatched vertices
                let right_free: Vec<Vertex> = (0..n)
                    .map_into::<Vertex>()
                    .filter(|v| !vertices.contains(v))
                    .collect();
                debug_assert_eq!(left_free.len(), right_free.len());

                let v = left_free.len();
                let prematch_rotated_shape = partial_fit.rotate_rotor(&shape_coordinates.matrix.clone());
                let prematch_rotated_shape = StrongPoints::<Vertex>::new(prematch_rotated_shape);

                if cfg!(debug_assertions) {
                    let partial_msd: f64 = vertices.iter()
                        .enumerate()
                        .map(|(i, j)| (cloud.matrix.column(i) - prematch_rotated_shape.column(*j)).norm_squared())
                        .sum();
                    approx::assert_relative_eq!(partial_fit.msd, partial_msd, epsilon=1e-6);
                }

                let cost_fn = |i, j| (cloud.column(left_free[i]) - prematch_rotated_shape.column(right_free[j])).norm_squared();
                let sub_permutation = match LAP_JV {
                    true => linear_assignment::optimal(v, &cost_fn),
                    false => linear_assignment::brute_force(v, &cost_fn)
                };

                // Fuse pre-match and best subpermutation
                for (i, j) in sub_permutation.iter_pairs() {
                    partial_map.insert(left_free[i], right_free[*j]);
                }

                if cfg!(debug_assertions) {
                    assert_eq!(partial_map.len(), n);
                    assert!(itertools::equal(
                        partial_map.keys().copied().sorted(), 
                        (0..n).map_into::<Column>()
                    ));
                    assert!(itertools::equal(
                        partial_map.values().copied().sorted(), 
                        (0..n).map_into::<Vertex>()
                    ));
                }

                // Make a clean quaternion fit with the full mapping
                let full_fit = cloud.quaternion_fit_with_map(&shape_coordinates, &partial_map);
                if full_fit.msd < best.msd {
                    PartialMsd {msd: full_fit.msd, mapping: partial_map}
                } else {
                    best
                }
            }
        );

    /* Given the best permutation for the rotational fit, we still have to
     * minimize over the isotropic scaling factor. It is cheaper to reorder the
     * positions once here for the minimization so that memory access is in-order
     * during the repeated scaling minimization function calls.
     *
     * NOTE: The bijection gets inverted here
     */
    let best_bijection = {
        let mut p = Permutation::identity(n);
        narrow.mapping.iter().for_each(|(c, v)| { p.sigma[v.get()] = c.get(); });
        Occupation::new(p)
    };

    if cfg!(debug_assertions) {
        for (c, v) in narrow.mapping.iter() {
            assert_eq!(best_bijection.get(v), Some(*c));
        }
    }

    let permuted_shape = shape_coordinates.apply_bijection(&best_bijection);
    let fit = cloud.quaternion_fit_with_rotor(&permuted_shape);
    let rotated_shape = fit.rotate_rotor(&permuted_shape.matrix);

    let csm = scaling::minimize_csm(&cloud.matrix, &rotated_shape);
    Ok(Similarity {bijection: best_bijection, csm})
}

pub fn polyhedron(x: Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    polyhedron_base::<5, true, true>(x, s)
}

#[cfg(test)]
mod tests {
    use crate::shapes::*;
    use crate::shapes::similarity::*;
    use crate::shapes::Matrix3N;
    use crate::quaternions::random_rotation;
    use crate::strong::bijection::bijections;

    fn random_cloud(n: usize) -> Matrix3N {
        unit_sphere_normalize(Matrix3N::new_random(n))
    }

    #[test]
    fn column_permutation() {
        let n = 6;
        let permutation = Permutation::random(n);

        let stator = random_cloud(n);
        let permuted = apply_permutation(&stator, &permutation);

        for i in 0..n {
            assert_eq!(stator.column(i), permuted.column(permutation[i]));
        }

        let reconstituted = apply_permutation(&permuted, &permutation.inverse());
        approx::assert_relative_eq!(stator, reconstituted);
    }

    #[test]
    fn normalization() {
        let cloud = random_cloud(6);

        // No centroid
        let centroid_norm = cloud.column_mean().norm();
        assert!(centroid_norm < 1e-6);

        // Longest vector is a unit vector
        let longest_norm = cloud.column_iter().map(|v| v.norm()).max_by(|a, b| a.partial_cmp(b).expect("Encountered NaNs")).unwrap();
        approx::assert_relative_eq!(longest_norm, 1.0);
    }

    #[test]
    fn scaling_minimization() {
        let cloud = random_cloud(6);
        let factor = 0.5 + 0.5 * rand::random::<f64>();
        assert!(factor <= 1.0);

        let result = scaling::minimize(&cloud.scale(factor), &cloud);
        approx::assert_relative_eq!(factor, result.factor);
        approx::assert_relative_eq!(0.0, result.msd);
    }

    #[test]
    fn scaling_direct() {
        let cloud = random_cloud(6);
        let factor = 0.5 + 0.5 * rand::random::<f64>();
        assert!(factor <= 1.0);

        approx::assert_relative_eq!(factor, scaling::direct_factor(&cloud.scale(factor), &cloud));
    }

    struct Case {
        shape_name: Name,
        bijection: Bijection<Vertex, Column>,
        cloud: StrongPoints<Column>
    }

    impl Case {
        fn random_shape_rotation(shape: &Shape) -> StrongPoints<Vertex> {
            let shape_coords = shape.coordinates.clone().insert_column(shape.size(), 0.0);
            let rotation = random_rotation().to_rotation_matrix();
            StrongPoints::new(unit_sphere_normalize(rotation * shape_coords))
        }

        fn random(shape: &Shape) -> Case {
            let bijection = {
                let mut p = Permutation::random(shape.size());
                p.sigma.push(p.set_size());
                Bijection::new(p)
            };

            let cloud = Self::random_shape_rotation(shape).apply_bijection(&bijection);
            Case { shape_name: shape.name, bijection, cloud }
        }

        fn indexed(shape: &Shape, index: usize) -> Case {
            let bijection = {
                let mut p = Permutation::from_index(shape.size(), index);
                p.sigma.push(p.set_size());
                Bijection::new(p)
            };

            let cloud = Self::random_shape_rotation(shape).apply_bijection(&bijection);
            Case { shape_name: shape.name, bijection, cloud }
        }

        fn can_find_bijection_with(&self, f: &dyn Fn(Matrix3N, Name) -> Result<Similarity, SimilarityError>) -> bool {
            let similarity = f(self.cloud.matrix.clone(), self.shape_name).expect("Fine");

            // Found bijection must be a rotation of the original bijection
            let shape = shape_from_name(self.shape_name);
            let rotations = shape.generate_rotations();
            let reference_bijection = pop_centroid(self.bijection.clone());
            let found_bijection = pop_centroid(similarity.bijection);


            let is_rotation_of_reference = Shape::is_rotation(&reference_bijection, &found_bijection, &rotations);
            let zero_csm = similarity.csm < 1e-6;

            is_rotation_of_reference && zero_csm
        }
    }

    fn pop_centroid<T>(mut p: Bijection<Vertex, T>) -> Bijection<Vertex, T> where T: NewTypeIndex {
        let maybe_centroid = p.permutation.sigma.pop();
        // Ensure centroid was at end of permutation
        assert_eq!(maybe_centroid, Some(p.set_size()));
        p
    }

    struct SimilarityFnTestBounds<'a> {
        f: &'a dyn Fn(Matrix3N, Name) -> Result<Similarity, SimilarityError>,
        max: usize,
        name: &'static str
    }

    fn similarity_fn_bounds() -> Vec<SimilarityFnTestBounds<'static>> {
        if cfg!(debug_assertions) {
            [
                SimilarityFnTestBounds {f: &polyhedron_reference, max: 5, name: "polyhedron reference"},
                SimilarityFnTestBounds {f: &polyhedron, max: 6, name: "polyhedron"}
            ].into()
        } else {
            [
                SimilarityFnTestBounds {f: &polyhedron_reference, max: 7, name: "polyhedron reference"},
                SimilarityFnTestBounds {f: &polyhedron, max: 8, name: "polyhedron"}
            ].into()
        }
    }

    const MAX_SHAPE_SIZE_EXHAUSTIVE: usize = if cfg!(debug_assertions) { 4 } else { 6 };

    #[test]
    fn polyhedron_reference_all_bijections() {
        for shape in SHAPES.iter().filter(|s| s.size() <= MAX_SHAPE_SIZE_EXHAUSTIVE) {
            let index_bound = Permutation::group_order(shape.size());
            for i in 0..index_bound {
                let case = Case::indexed(shape, i);
                if !case.can_find_bijection_with(&polyhedron_reference) {
                    panic!("Couldn't find {} in {} with polyhedron reference", case.bijection, shape.name);
                }
            }
        }
    }

    #[test]
    fn polyhedron_all_bijections() {
        for shape in SHAPES.iter().filter(|s| s.size() <= MAX_SHAPE_SIZE_EXHAUSTIVE) {
            let index_bound = Permutation::group_order(shape.size());
            for i in 0..index_bound {
                let case = Case::indexed(shape, i);
                if !case.can_find_bijection_with(&polyhedron) {
                    panic!("Couldn't find {} in {} with polyhedron", case.bijection, shape.name);
                }
            }
        }
    }

    #[test]
    fn stochastic_similarities_tests() {
        for shape in SHAPES.iter().filter(|s| s.size() > MAX_SHAPE_SIZE_EXHAUSTIVE) {
            let shape_size = shape.size();
            let case = Case::random(shape);

            for bounds in similarity_fn_bounds() {
                if shape_size <= bounds.max && !case.can_find_bijection_with(bounds.f) {
                    panic!("Couldn't find {} in {} with {}", case.bijection, shape.name, bounds.name);
                }
            }
        }
    }

    #[test]
    fn skips_recoverable_by_rotation() {
        for shape in SHAPES.iter().filter(|s| s.size() < 7) {
            let rotations = shape.generate_rotations();

            let mut recovered_bijections = HashSet::new();
            for rotationally_unique in SkipsBijectionGenerator::new(shape.name) {
                // Need to invert since the centroid is only last in vertex -> column
                // and need to remove it to rotate in shape vertex space
                let mut inverted = rotationally_unique.inverse();
                let centroid_vertex = inverted.permutation.sigma.pop().unwrap();
                for rotation in rotations.iter() {
                    let mut rotated = rotation.compose(&inverted).unwrap();
                    rotated.permutation.sigma.push(centroid_vertex);

                    // no need to re-invert
                    recovered_bijections.insert(rotated);
                }
            }

            assert_eq!(recovered_bijections.len(), Permutation::group_order(shape.size() + 1));
        }
    }

    #[test]
    fn skip_generator() {
        for shape in SHAPES.iter().filter(|s| s.size() < 7) {
            let skips = skip_vertices(shape.name);

            itertools::assert_equal(
                bijections(shape.size() + 1)
                    .filter(|b| !skips[(b.permutation[0], b.permutation[1])]),
                SkipsBijectionGenerator::new(shape.name)
            );
        }
    }
}
