// TODO
// - Rewrite polyhedron similarity functions with strong permutation
// - Understand test failure (could be either buggy or logically faulty)
// - Expand shape data
// - Add lapjv linear assignment variation
// - Unify implementations (?)
// - "Horn [13] considered including a scaling in the transfor-
//   mation T in eq. (11). This is quite easily accommodated." 
//   from https://arxiv.org/pdf/physics/0506177.pdf
//   -> Could maybe radically simplify the scaling optimization step
// 


extern crate nalgebra as na;

pub type Matrix3N = na::Matrix3xX<f64>;
pub type Matrix3 = na::Matrix3<f64>;

pub type Quaternion = na::UnitQuaternion<f64>;

extern crate thiserror;
use thiserror::Error;

extern crate argmin;

use std::collections::{HashMap, HashSet};
use itertools::Itertools;

use derive_more::{From, Into};
use crate::index::{Index, NewTypeIndex};

#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub enum Name {
    // 2
    Line,
    Bent,
    // 3
    EquilateralTriangle,
    VacantTetrahedron,
    T,
    // 4
    Tetrahedron,
    Square,
    Seesaw,
    TrigonalPyramid,
    // 5
    SquarePyramid,
    TrigonalBipyramid,
    Pentagon,
    // 6
    // Octahedron,
    // TrigonalPrism,
    // PentagonalPyramid,
    // Hexagon
}

impl Name {
    pub fn repr(&self) -> &'static str {
        match self {
            Name::Line => "line",
            Name::Bent => "bent",
            Name::EquilateralTriangle => "triangle",
            Name::VacantTetrahedron => "vacant tetrahedron",
            Name::T => "T-shaped",
            Name::Tetrahedron => "tetrahedron",
            Name::Square => "square",
            Name::Seesaw => "seesaw",
            Name::TrigonalPyramid => "trigonal pyramid",
            Name::SquarePyramid => "square pyramid",
            Name::TrigonalBipyramid => "trigonal bipyramid",
            Name::Pentagon => "pentagon",
        }
    }
}

use crate::permutation::{Permutation, permutations};
use crate::bijection::{Bijection, bijections};
use crate::quaternions;
use crate::quaternions::Fit;

#[derive(Index, From, Into, Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Vertex(u8);

type Rotation = Bijection<Vertex, Vertex>;
type Mirror = Bijection<Vertex, Vertex>;

#[derive(Index, From, Into, Debug, Copy, Clone, PartialEq, Eq)]
pub struct Column(u8);

pub static ORIGIN: u8 = u8::MAX;

pub struct Shape {
    pub name: Name,
    /// Unit sphere coordinates without a centroid
    pub coordinates: Matrix3N,
    /// Spatial rotational basis expressed by vertex permutations
    pub rotation_basis: Vec<Rotation>,
    /// Minimal set of tetrahedra required to distinguish volumes in DG
    pub tetrahedra: Vec<[u8; 4]>,
    /// Mirror symmetry element expressed by vertex permutation, if present
    pub mirror: Option<Mirror>
}

impl Shape {
    /// Number of vertices of the shape
    pub fn size(&self) -> usize {
        self.coordinates.ncols()
    }

    /// Generate a full set of rotations from a shape's rotational basis
    ///
    /// ```
    /// # use molassembler::shapes::*;
    /// # use molassembler::bijection::bijections;
    /// # use std::collections::HashSet;
    /// # use std::iter::FromIterator;
    /// let line_rotations = LINE.generate_rotations();
    /// assert_eq!(line_rotations, HashSet::from_iter(bijections(2)));
    /// assert!(line_rotations.iter().all(|r| r.set_size() == 2));
    ///
    /// let tetrahedron_rotations = TETRAHEDRON.generate_rotations();
    /// assert_eq!(tetrahedron_rotations.len(), 12);
    /// assert!(tetrahedron_rotations.iter().all(|r| r.set_size() == 4));
    ///
    /// ```
    pub fn generate_rotations(&self) -> HashSet<Rotation> {
        let mut rotations: HashSet<Rotation> = HashSet::new();
        rotations.insert(Rotation::identity(self.size()));
        let max_basis_idx = self.rotation_basis.len() - 1;

        struct Frame {
            permutation: Rotation,
            next_basis: usize
        }

        let mut stack = Vec::<Frame>::new();
        stack.push(Frame {permutation: Rotation::identity(self.size()), next_basis: 0});

        // Tree-like traversal, while tracking rotations applied to get new rotations and pruning
        // if rotations have been seen before
        while stack.first().unwrap().next_basis <= max_basis_idx {
            let latest = stack.last().unwrap();
            let next_rotation = &self.rotation_basis[latest.next_basis];
            let generated = latest.permutation.compose(next_rotation).unwrap();

            if rotations.insert(generated.clone()) {
                // Continue finding new things from this structure
                stack.push(Frame {permutation: generated, next_basis: 0});
            } else {
                // Try to pop unincrementable stack frames
                while stack.len() > 1 && stack.last().unwrap().next_basis == max_basis_idx {
                    stack.pop();
                }

                stack.last_mut().unwrap().next_basis += 1;
            }
        }

        rotations
    }

    // maybe strong variant is a, b of type X -> Vertex
    pub fn is_rotation(&self, a: &Permutation, b: &Permutation, rotations: &HashSet<Rotation>) -> bool {
        rotations.iter().any(|r| a.compose(&r.permutation).expect("Passed bad rotations") == *b)
    }
}

fn make_rotation(slice: &[u8]) -> Rotation {
    Rotation::new(Permutation {sigma: slice.to_vec()})
}

fn make_mirror(slice: &[u8]) -> Option<Mirror> {
    Some(make_rotation(slice))
}

lazy_static! {
    pub static ref LINE: Shape = Shape {
        name: Name::Line,
        coordinates: Matrix3N::from_column_slice(&[
             1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref BENT: Shape = Shape {
        name: Name::Bent,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.292372, 0.956305, 0.0
        ]),
        rotation_basis: vec![make_rotation(&[1, 0])],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref EQUILATERAL_TRIANGLE: Shape = Shape {
        name: Name::EquilateralTriangle,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[1, 2, 0]),
            make_rotation(&[0, 2, 1])
        ],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref VACANT_TETRAHEDRON: Shape = Shape {
        name: Name::VacantTetrahedron,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, -0.366501, 0.930418,
            0.805765, -0.366501, -0.465209,
            -0.805765, -0.366501, -0.465209
        ]),
        rotation_basis: vec![make_rotation(&[2, 0, 1])],
        tetrahedra: vec![[ORIGIN, 0, 1, 2]],
        mirror: make_mirror(&[0, 2, 1])
    };

    pub static ref TSHAPE: Shape = Shape {
        name: Name::T,
        coordinates: Matrix3N::from_column_slice(&[
            -1.0, -0.0, -0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]),
        rotation_basis: vec![Rotation::new(Permutation {sigma: vec![2, 1, 0]})],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref TETRAHEDRON: Shape = Shape {
        name: Name::Tetrahedron,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, 1.0, 0.0,
            0.0, -0.333807, 0.942641,
            0.816351, -0.333807, -0.471321,
            -0.816351, -0.333807, -0.471321
        ]),
        rotation_basis: vec![
            make_rotation(&[0, 3, 1, 2]),
            make_rotation(&[2, 1, 3, 0]),
            make_rotation(&[3, 0, 2, 1]),
            make_rotation(&[1, 2, 0, 3])
        ],
        tetrahedra: vec![[0, 1, 2, 3]],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    pub static ref SQUARE: Shape = Shape {
        name: Name::Square,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, -0.0, -0.0,
            -0.0, -1.0, -0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[3, 0, 1, 2]),
            make_rotation(&[1, 0, 3, 2]),
            make_rotation(&[3, 2, 1, 0]),
        ],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref SEESAW: Shape = Shape {
        name: Name::Seesaw,
        coordinates: Matrix3N::from_column_slice(&[
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
            -0.5, 0.0, -0.866025,
            -0.0, -1.0, -0.0
        ]),
        rotation_basis: vec![make_rotation(&[3, 2, 1, 0])],
        tetrahedra: vec![[0, ORIGIN, 1, 2], [ORIGIN, 3, 1, 2]],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    pub static ref TRIGONALPYRAMID: Shape = Shape {
        name: Name::TrigonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0,
            0.0, 0.0, 1.0
        ]),
        rotation_basis: vec![make_rotation(&[2, 0, 1, 3])],
        tetrahedra: vec![[0, 1, 3, 2]],
        mirror: make_mirror(&[0, 2, 1, 3])
    };

    pub static ref SQUAREPYRAMID: Shape = Shape {
        name: Name::SquarePyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            -1.0, 0.0, 0.0,
            0.0, -1.0, 0.0,
            0.0, 0.0, 1.0,
        ]),
        rotation_basis: vec![make_rotation(&[3, 0, 1, 2, 4])],
        tetrahedra: vec![
            [0, 1, 4, ORIGIN],
            [1, 2, 4, ORIGIN],
            [2, 3, 4, ORIGIN],
            [3, 0, 4, ORIGIN],
        ],
        mirror: make_mirror(&[1, 0, 3, 2, 4])
    };

    pub static ref TRIGONALBIPYRAMID: Shape = Shape {
        name: Name::TrigonalBipyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, -1.0
        ]),
        rotation_basis: vec![
            make_rotation(&[2, 0, 1, 3, 4]), // C3
            make_rotation(&[0, 2, 1, 4, 3]), // C2 on 0
        ],
        tetrahedra: vec![[0, 1, 3, 2], [0, 1, 2, 4]],
        mirror: make_mirror(&[0, 2, 1, 3, 4])
    };

    pub static ref PENTAGON: Shape = Shape {
        name: Name::Pentagon,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            0.309017, 0.951057, 0.0,
            -0.809017, 0.587785, 0.0,
            -0.809017, -0.587785, 0.0,
            0.309017, -0.951057, 0.0
        ]),
        rotation_basis: vec![
            make_rotation(&[4, 0, 1, 2, 3]),
            make_rotation(&[0, 4, 3, 2, 1]),
        ],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT, &EQUILATERAL_TRIANGLE, &VACANT_TETRAHEDRON, &TSHAPE, &TETRAHEDRON, &SQUARE, &SEESAW, &TRIGONALPYRAMID, &SQUAREPYRAMID, &TRIGONALBIPYRAMID, &PENTAGON];
}

pub fn shape_from_name(name: Name) -> &'static Shape {
    let shape = SHAPES[name as usize];
    assert_eq!(shape.name, name);
    shape
}

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

struct StrongPoints<IndexingType> where IndexingType: NewTypeIndex {
    pub matrix: Matrix3N,
    index_type: PhantomData<IndexingType>
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum SimilarityError {
    #[error("Number of positions does not match shape size")]
    PositionNumberMismatch
}

pub struct Similarity {
    pub bijection: Bijection<Column, Vertex>,
    pub csm: f64,
}

/// Polyhedron similarity performing quaternion fits on all vertices 
pub fn polyhedron_similarity(x: &Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    type Occupation = Bijection<Column, Vertex>;

    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let cloud = unit_sphere_normalize(x.clone());
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);

    let evaluate_permutation = |p: Occupation| -> (Occupation, Fit) {
        let permuted_shape = apply_permutation(&shape_coordinates, &p.permutation);
        let fit = quaternions::fit(&cloud, &permuted_shape);
        (p, fit)
    };

    let (best_bijection, best_fit) = bijections(n)
        .map(evaluate_permutation)
        .min_by(|(_, fit_a), (_, fit_b)| fit_a.msd.partial_cmp(&fit_b.msd).expect("NaN in MSDs"))
        .expect("Not a single permutation available to try");

    let permuted_shape = apply_permutation(&shape_coordinates, &best_bijection.permutation);
    let rotated_shape = best_fit.rotate_rotor(permuted_shape);

    let csm = scaling::minimize(&cloud, &rotated_shape);
    if cfg!(debug_assertions) {
        let direct = scaling::direct(&cloud, &rotated_shape);
        if (direct - csm).abs() > 1e-6 {
            println!("- minimization: {:e}, direct: {:e}", csm, direct);
        }
    }
    Ok(Similarity {bijection: best_bijection, csm})
}

/// Polyhedron similarity performing quaternion fits only on a limited number of 
/// vertices before assigning the rest by brute force linear assignment
pub fn polyhedron_similarity_shortcut(x: &Matrix3N, s: Name) -> Result<Similarity, SimilarityError> {
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let n_prematch = 5;
    if n <= n_prematch {
        return polyhedron_similarity(x, s);
    }

    let cloud = unit_sphere_normalize(x.clone());
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);

    type PartialPermutation = HashMap<usize, usize>;

    struct PartialMsd {
        msd: f64,
        mapping: PartialPermutation,
    }
    let left_free: Vec<usize> = (n_prematch..n).collect();

    let narrow = (0..n).permutations(n_prematch).fold(
        PartialMsd {msd: f64::MAX, mapping: PartialPermutation::new()},
        |best, vertices| -> PartialMsd {
            let mut partial_map = PartialPermutation::with_capacity(n);
            vertices.iter().enumerate().for_each(|(k, v)| { partial_map.insert(k, *v); });
            let partial_fit = quaternions::fit_with_map(&cloud, &shape_coordinates, &partial_map); 

            // If the msd caused only by the partial map is already worse, skip
            if partial_fit.msd > best.msd {
                return best;
            }

            // Assemble cost matrix of matching each pair of unmatched vertices
            let right_free: Vec<usize> = (0..n)
                .filter(|i| !vertices.contains(i))
                .collect();
            debug_assert_eq!(left_free.len(), right_free.len());

            let v = left_free.len();
            let prematch_rotated_shape = partial_fit.rotate_rotor(shape_coordinates.clone());

            if cfg!(debug_assertions) {
                let partial_msd: f64 = vertices.iter()
                    .enumerate()
                    .map(|(i, j)| (cloud.column(i) - prematch_rotated_shape.column(*j)).norm_squared())
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
                assert!(itertools::equal(partial_map.keys().copied().sorted(), 0..n));
                assert!(itertools::equal(partial_map.values().copied().sorted(), 0..n));
            }

            // Make a clean quaternion fit with the full mapping
            let full_fit = quaternions::fit_with_map(&cloud, &shape_coordinates, &partial_map);

            println!("Tried vertices {:?}, linear assigning {:?}, msd {:e}", vertices, right_free, full_fit.msd);

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
    let best_permutation = {
        let mut p = Permutation::identity(n);
        narrow.mapping.iter().for_each(|(i, j)| { p.sigma[*i] = *j as u8; });
        p
    };

    if cfg!(debug_assertions) {
        for (i, j) in narrow.mapping.iter() {
            assert_eq!(best_permutation[*i] as usize, *j);
        }
    }

    let permuted_shape = apply_permutation(&shape_coordinates, &best_permutation);
    let fit = quaternions::fit(&cloud, &permuted_shape);
    let rotated_shape = fit.rotate_rotor(permuted_shape);

    let csm = scaling::minimize(&cloud, &rotated_shape);
    println!("- minimization: {:e}, direct: {:e}", csm, scaling::direct(&cloud, &rotated_shape));
    Ok(Similarity {bijection: Bijection::new(best_permutation), csm})
}

#[cfg(test)]
mod tests {
    use crate::shapes::*;
    use nalgebra::{Vector3, Unit};

    fn random_discrete(n: usize) -> usize {
        let float = rand::random::<f32>();
        (float * n as f32) as usize
    }

    fn random_permutation(n: usize) -> Permutation {
        Permutation::from_index(n, random_discrete(Permutation::group_order(n)))
    }

    fn random_rotation() -> Quaternion {
        let random_axis = Unit::new_normalize(Vector3::new_random());
        let random_angle = rand::random::<f64>() * std::f64::consts::PI;
        Quaternion::from_axis_angle(&random_axis, random_angle)
    }

    fn random_cloud(n: usize) -> Matrix3N {
        unit_sphere_normalize(Matrix3N::new_random(n))
    }

    #[test]
    fn column_permutation() {
        let n = 6;
        let permutation_index = random_discrete(Permutation::group_order(n));
        let permutation = Permutation::from_index(n, permutation_index);

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
        let perm = Permutation::from_index(4, 23);
        for rot in &tetr_rotations {
            let rotated_perm = perm.compose(&rot.permutation).expect("fine");
            assert!(TETRAHEDRON.is_rotation(&perm, &rotated_perm, &tetr_rotations));
        }
    }

    #[test]
    fn similarity_postconditions() {
        for shape in SHAPES.iter() {
            let cloud = shape.coordinates.clone().insert_column(shape.size(), 0.0);

            let random_rotation = random_rotation().to_rotation_matrix();
            let rotated_cloud = unit_sphere_normalize(random_rotation * cloud);
            // Centroid preserving random permutation
            let random_permutation = {
                let mut p = random_permutation(rotated_cloud.ncols() - 1);
                p.sigma.push(p.set_size() as u8);
                p
            };
            let permuted_cloud = apply_permutation(&rotated_cloud, &random_permutation);

            let similarity = polyhedron_similarity(&permuted_cloud, shape.name).expect("Ok");

            // Bijection must be a rotation of applied random permutation
            let shape_rotations = shape.generate_rotations();
            let pop_centroid = |p: &Permutation| {
                let mut q = p.clone();
                let maybe_centroid = q.sigma.pop();
                // Ensure centroid was at end of permutation
                assert_eq!(maybe_centroid, Some(q.set_size() as u8));
                q
            };
            let a = pop_centroid(&random_permutation);
            let b = pop_centroid(&similarity.bijection.permutation);
            println!("shape {}, random: {}, result: {}", shape.name.repr(), a, b);
            assert!(shape.is_rotation(&a, &b, &shape_rotations));
        }
    }

    // Test variations on the initial implementation by ensuring their results
    // are rotations of one another
    #[test]
    fn self_similarity() {
        let fns = [polyhedron_similarity, polyhedron_similarity_shortcut];
        let fn_names = ["similarity", "similarity_shortcut"];

        for shape in SHAPES.iter() {
            if shape.size() < 5 {
                continue;
            }

            let cloud = shape.coordinates.clone().insert_column(shape.size(), 0.0);
            let random_rotation = random_rotation().to_rotation_matrix();
            let rotated_cloud = unit_sphere_normalize(random_rotation * cloud);
            // Centroid preserving random permutation
            let random_permutation = {
                let mut p = random_permutation(rotated_cloud.ncols() - 1);
                p.sigma.push(p.set_size() as u8);
                p
            };
            let permuted_cloud = apply_permutation(&rotated_cloud, &random_permutation);

            let results: Vec<Similarity> = fns.iter().map(|f| f(&permuted_cloud, shape.name).expect("Fine")).collect();

            for (fn_name, similarity) in fn_names.iter().zip(&results) {
                println!("Algorithm {} achieved similarity {:e} of {} with permutation {}", fn_name, similarity.csm, shape.name.repr(), similarity.bijection);

                println!("true {} <-> minimal {}, same: {}, inverse: {}", random_permutation, similarity.bijection, random_permutation == similarity.bijection.permutation, random_permutation == similarity.bijection.permutation.inverse());
                assert!(similarity.csm < 1e-6);
                // Centroid must be last
                assert_eq!(*similarity.bijection.permutation.sigma.last().unwrap(), shape.size() as u8);
            }

            // Found permutations are rotations of one another
            let shape_rotations = shape.generate_rotations();
            let fn_permutation_results: Vec<Permutation> = results.iter().map(|s| { let mut q = s.bijection.permutation.clone(); q.sigma.pop(); q }).collect(); // drop centroid
            let mutual_rotations = fn_permutation_results.iter().tuple_windows().all(|(p, q)| shape.is_rotation(p, q, &shape_rotations));
            assert!(mutual_rotations);

            println!("");
        }
    }

    // for v in cloud.column_iter_mut() {
    //     v += Vector3::new_random().normalize().scale(0.1);
    // }
}
