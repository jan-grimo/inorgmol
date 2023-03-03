// TODO
// - Add lapjv linear assignment variation
// - Add skip lists
// - Unify implementations (?)
// - "Horn [13] considered including a scaling in the transfor-
//   mation T in eq. (11). This is quite easily accommodated." 
//   from https://arxiv.org/pdf/physics/0506177.pdf
//   -> Could maybe radically simplify the scaling optimization step
// 

extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use thiserror::Error;
use memoize::memoize;
use itertools::Itertools;
use std::collections::HashMap;

use crate::strong::matrix::StrongPoints;
use crate::strong::bijection::{Bijection, bijections};
use crate::permutation::{Permutation, permutations};
use crate::quaternions::Fit;
use crate::shapes::*;

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

// Column-permute a matrix
//
// Post-condition is x.column(i) == result.column(p(i))
pub fn apply_permutation(x: &Matrix3N, p: &Permutation) -> Matrix3N {
    assert_eq!(x.ncols(), p.set_size());
    let inverse = p.inverse();
    Matrix3N::from_fn(x.ncols(), |i, j| x[(i, inverse[j] as usize)])
}

mod scaling {
    use crate::shapes::Matrix3N;

    use argmin::core::{CostFunction, Error, Executor};
    use argmin::solver::goldensectionsearch::GoldenSectionSearch;

    fn evaluate_msd(cloud: &Matrix3N, rotated_shape: &Matrix3N, factor: f64) -> f64 {
        (cloud.clone() - rotated_shape.scale(factor))
            .column_iter()
            .map(|col| col.norm_squared())
            .sum()
    }

    struct CoordinateScalingProblem<'a> {
        pub cloud: &'a Matrix3N,
        pub rotated_shape: &'a Matrix3N
    }

    impl CostFunction for CoordinateScalingProblem<'_> {
        type Param = f64;
        type Output = f64;

        fn cost(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            Ok(evaluate_msd(&self.cloud, &self.rotated_shape, *p))
        }
    }

    fn normalize(cloud: &Matrix3N, msd: f64) -> f64 {
        100.0 * msd / cloud.column_iter().map(|v| v.norm_squared()).sum::<f64>()
    }

    pub fn minimize(cloud: &Matrix3N, shape: &Matrix3N) -> f64 {
        let phi_sectioning = GoldenSectionSearch::new(0.5, 1.1).expect("yup");
        let scaling_problem = CoordinateScalingProblem {
            cloud,
            rotated_shape: shape 
        };
        let result = Executor::new(scaling_problem, phi_sectioning)
            .configure(|state| state.param(1.0).max_iters(100))
            .run()
            .unwrap();

        if cfg!(debug_assertions) {
            let scale = result.state.best_param.expect("present");
            let direct = (cloud.column_iter().map(|v| v.norm_squared()).sum::<f64>()
                / shape.column_iter().map(|v| v.norm_squared()).sum::<f64>()).sqrt();
            if (scale - direct).abs() > 1e-6 {
                println!("-- Scaling {:e} vs direct {:e}", scale, direct);
            }
        }

        normalize(&cloud, result.state.best_cost)
    }

    // TODO Untested, maybe faster?
    // Final scale formula of section 2E in "Closed-form solution of absolute orientation with
    // quaternions" by Berthold K.P. Horn in J. Opt. Soc. Am. A, 1986
    pub fn direct(cloud: &Matrix3N, shape: &Matrix3N) -> f64 {
        let factor = (cloud.column_iter().map(|v| v.norm_squared()).sum::<f64>()
            / shape.column_iter().map(|v| v.norm_squared()).sum::<f64>()).sqrt();
        normalize(&cloud, evaluate_msd(&cloud, &shape, factor))
    }
}

// Yields a boolean matrix indicating
#[memoize]
pub fn skip_vertices(shape_name: Name) -> na::DMatrix<bool> {
    let shape = shape_from_name(shape_name);

    let s = shape.size();
    let mut skips = na::DMatrix::<bool>::from_element(s + 1, s + 1, true);
    let rotations = shape.generate_rotations();
    let viable_vertices: Vec<Vertex> = shape.vertex_groups().iter().map(|g| *g.first().unwrap()).collect();
    for i in viable_vertices.iter() {
        let vertex_groups = shape.vertex_groups_holding(*i, &rotations);
        for group in vertex_groups {
            let j = group.first().expect("Vertex groups shouldn't be empty");
            skips[(i.get() as usize, j.get() as usize)] = false;
        }

        // Centroid is always a valid second match
        skips[(i.get() as usize, s)] = false;
    }

    // If matching centroid first, reuse viable_vertices for second match
    for i in 0..s {
        skips[(s, i)] = skips[(i, 0)];
    }

    skips
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SimilarityError {
    #[error("Number of positions does not match shape size")]
    PositionNumberMismatch
}

pub struct Similarity {
    pub bijection: Bijection<Vertex, Column>,
    pub csm: f64,
}

/// Polyhedron similarity performing quaternion fits on all rotationally distinct bijections
pub fn polyhedron_reference(x: &Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    type Occupation = Bijection<Vertex, Column>;

    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let cloud = StrongPoints::new(unit_sphere_normalize(x.clone()));
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = StrongPoints::new(unit_sphere_normalize(shape_coordinates));

    let evaluate_permutation = |p: Occupation| -> (Occupation, Fit) {
        let permuted_shape = shape_coordinates.apply_bijection(&p);
        let fit = cloud.quaternion_fit_with_rotor(&permuted_shape);
        (p, fit)
    };

    let skips = skip_vertices(s);

    let (best_bijection, best_fit) = bijections(n)
        .filter(|b| skips[(b.permutation[0] as usize, b.permutation[1] as usize)])
        .map(evaluate_permutation)
        .min_by(|(_, fit_a), (_, fit_b)| fit_a.msd.partial_cmp(&fit_b.msd).expect("NaN in MSDs"))
        .expect("Not a single permutation available to try");

    let permuted_shape = shape_coordinates.apply_bijection(&best_bijection);
    let rotated_shape = best_fit.rotate_rotor(&permuted_shape.matrix);

    let csm = scaling::minimize(&cloud.matrix, &rotated_shape);
    if cfg!(debug_assertions) {
        let direct = scaling::direct(&cloud.matrix, &rotated_shape);
        if (direct - csm).abs() > 1e-6 {
            println!("- minimization: {:e}, direct: {:e}", csm, direct);
        }
    }
    Ok(Similarity {bijection: best_bijection, csm})
}

/// Polyhedron similarity performing quaternion fits only on a limited number of 
/// vertices before assigning the rest by brute force linear assignment
pub fn polyhedron(x: &Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    type Occupation = Bijection<Vertex, Column>;
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let n_prematch = 5;
    if n <= n_prematch {
        return polyhedron_reference(x, s);
    }

    let cloud = StrongPoints::<Column>::new(unit_sphere_normalize(x.clone()));
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
    let left_free: Vec<Column> = (n_prematch..n).map(|i| Column::from(i as u8)).collect();
    let skips = skip_vertices(s);

    let narrow = (0..n)
        .map(|i| Vertex::from(i as u8))
        .permutations(n_prematch)
        .filter(|p| skips[(p[0].get() as usize, p[1].get() as usize)])
        .fold(
            PartialMsd {msd: f64::MAX, mapping: PartialPermutation::new()},
            |best, vertices| -> PartialMsd {
                let mut partial_map = PartialPermutation::with_capacity(n);
                vertices.iter().enumerate().for_each(|(c, v)| { partial_map.insert(Column::from(c as u8), *v); });
                let partial_fit = cloud.quaternion_fit_with_map(&shape_coordinates, &partial_map);

                // If the msd caused only by the partial map is already worse, skip
                if partial_fit.msd > best.msd {
                    return best;
                }

                // Assemble cost matrix of matching each pair of unmatched vertices
                let right_free: Vec<Vertex> = (0..n)
                    .map(|i| Vertex::from(i as u8))
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
                let costs = na::DMatrix::from_fn(v, v, cost_fn);
                
                // Brute-force the costs matrix linear assignment permutation
                let (_, sub_permutation) = permutations(v)
                    .map(|permutation| ((0..v).map(|i| costs[(i, permutation[i] as usize)]).sum(), permutation) )
                    .min_by(|(a, _): &(f64, Permutation), (b, _): &(f64, Permutation)| a.partial_cmp(b).expect("Encountered NaNs"))
                    .expect("At least one permutation present");

                // Fuse pre-match and best subpermutation
                for (i, j) in sub_permutation.iter_pairs() {
                    partial_map.insert(left_free[i], right_free[*j as usize]);
                }

                if cfg!(debug_assertions) {
                    assert_eq!(partial_map.len(), n);
                    assert!(itertools::equal(
                        partial_map.keys().copied().sorted(), 
                        (0..n).map(|i| Column::from(i as u8))
                    ));
                    assert!(itertools::equal(
                        partial_map.values().copied().sorted(), 
                        (0..n).map(|i| Vertex::from(i as u8))
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
     * during the repeated scaling minimization function call.
     */
    let best_bijection = {
        let mut p = Permutation::identity(n);
        narrow.mapping.iter().for_each(|(c, v)| { p.sigma[v.get() as usize] = c.get(); });
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

    let csm = scaling::minimize(&cloud.matrix, &rotated_shape);
    Ok(Similarity {bijection: best_bijection, csm})
}

#[cfg(test)]
mod tests {
    use crate::shapes::*;
    use crate::shapes::similarity::*;
    use crate::shapes::Matrix3N;
    use crate::quaternions::random_rotation;

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
            assert_eq!(stator.column(i), permuted.column(permutation[i] as usize));
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
    fn is_rotation_works() {
        let tetr_rotations = TETRAHEDRON.generate_rotations();
        let occupation: Bijection<Vertex, Column> = Bijection::from_index(4, 23);
        for rot in &tetr_rotations {
            let rotated_occupation = rot.compose(&occupation).expect("fine");
            assert!(TETRAHEDRON.is_rotation(&occupation, &rotated_occupation, &tetr_rotations));
        }
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
                p.sigma.push(p.set_size() as u8);
                Bijection::new(p)
            };

            let cloud = Self::random_shape_rotation(shape).apply_bijection(&bijection);
            Case { shape_name: shape.name, bijection, cloud }
        }

        fn indexed(shape: &Shape, index: usize) -> Case {
            let bijection = {
                let mut p = Permutation::from_index(shape.size(), index);
                p.sigma.push(p.set_size() as u8);
                Bijection::new(p)
            };

            let cloud = Self::random_shape_rotation(shape).apply_bijection(&bijection);
            Case { shape_name: shape.name, bijection, cloud }
        }

        fn assert_postconditions_with(&self, f: &dyn Fn(&Matrix3N, Name) -> Result<Similarity, SimilarityError>) {
            let similarity = f(&self.cloud.matrix, self.shape_name).expect("Fine");

            // Found bijection must be a rotation of the original bijection
            let shape = shape_from_name(self.shape_name);
            let shape_rotations = shape.generate_rotations();
            let reference_bijection = pop_centroid(self.bijection.clone());
            let found_bijection = pop_centroid(similarity.bijection);
            assert!(shape.is_rotation(&reference_bijection, &found_bijection, &shape_rotations));

            // Fit must be vanishingly small
            assert!(similarity.csm < 1e-6);
        }
    }

    fn pop_centroid<Key, Value>(mut p: Bijection<Key, Value>) -> Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
        let maybe_centroid = p.permutation.sigma.pop();
        // Ensure centroid was at end of permutation
        assert_eq!(maybe_centroid, Some(p.set_size() as u8));
        p
    }

    struct SimilarityFnTestBounds<'a> {
        f: &'a dyn Fn(&Matrix3N, Name) -> Result<Similarity, SimilarityError>,
        min: usize,
        max: usize,
    }

    fn similarity_fn_bounds() -> Vec<SimilarityFnTestBounds<'static>> {
        if cfg!(debug_assertions) {
            [
                SimilarityFnTestBounds {f: &polyhedron_reference, min: 1, max: 5},
                SimilarityFnTestBounds {f: &polyhedron, min: 1, max: 6}
            ].into()
        } else {
            [
                SimilarityFnTestBounds {f: &polyhedron_reference, min: 1, max: 7},
                SimilarityFnTestBounds {f: &polyhedron, min: 1, max: 8}
            ].into()
        }
    }

    #[test]
    fn try_all_similarities() {
        for shape in SHAPES.iter() {
            let shape_size = shape.size();
            let case = Case::random(shape);

            for bounds in similarity_fn_bounds() {
                if bounds.min <= shape_size && shape_size <= bounds.max {
                    case.assert_postconditions_with(bounds.f);
                }
            }
        }
    }

    #[test]
    fn pyramid_exhaustive() {
        let test_shape: &Shape = {
            if cfg!(debug_assertions) { &TRIGONALPYRAMID } else { &SQUAREPYRAMID }
        };

        let index_bound = Permutation::group_order(test_shape.size());
        for i in 0..index_bound {
            let case = Case::indexed(test_shape, i);

            for bounds in similarity_fn_bounds() {
                case.assert_postconditions_with(bounds.f);
            }
        }
    }

    // for v in cloud.column_iter_mut() {
    //     v += Vector3::new_random().normalize().scale(0.1);
    // }
}
