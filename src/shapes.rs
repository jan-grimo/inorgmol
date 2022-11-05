extern crate nalgebra as na;

pub type Matrix3N = na::Matrix3xX<f64>;
pub type Matrix3 = na::Matrix3<f64>;
type Matrix4 = na::Matrix4<f64>;

pub type Quaternion = na::UnitQuaternion<f64>;

extern crate thiserror;
use thiserror::Error;

extern crate argmin;

use std::collections::{HashMap, HashSet};
use itertools::Itertools;

use derive_more::{From, Into};
use crate::index::Index;

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
    // SquarePyramid,
    // TrigonalBipyramid,
    // Pentagon,
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
            Name::TrigonalPyramid => "trigonal pyramid"
        }
    }
}

use crate::permutation;
pub type Permutation = permutation::Permutation;

#[derive(Index, From, Into, Debug, Copy, Clone, PartialEq)]
pub struct Vertex(u8);

pub static ORIGIN: u8 = u8::MAX;

pub struct Shape {
    pub name: Name,
    /// Unit sphere coordinates without a centroid
    pub coordinates: Matrix3N,
    /// Spatial rotational basis expressed by vertex permutations
    pub rotation_basis: Vec<Permutation>,
    /// Minimal set of tetrahedra required to distinguish volumes in DG
    pub tetrahedra: Vec<[u8; 4]>,
    /// Mirror symmetry element expressed by vertex permutation, if present
    pub mirror: Option<Permutation>
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
    /// # use molassembler::permutation::*;
    /// # use std::collections::HashSet;
    /// # use std::iter::FromIterator;
    /// let line_rotations = LINE.generate_rotations();
    /// assert_eq!(line_rotations, HashSet::from_iter(permutations(2)));
    /// assert!(line_rotations.iter().all(|r| r.len() == 2));
    ///
    /// let tetrahedron_rotations = TETRAHEDRON.generate_rotations();
    /// assert_eq!(tetrahedron_rotations.len(), 12);
    /// assert!(tetrahedron_rotations.iter().all(|r| r.len() == 4));
    ///
    /// ```
    pub fn generate_rotations(&self) -> HashSet<Permutation> {
        let mut rotations: HashSet<Permutation> = HashSet::new();
        rotations.insert(Permutation::identity(self.size()));
        let max_basis_idx = self.rotation_basis.len() - 1;

        struct Frame {
            permutation: Permutation,
            next_basis: usize
        }

        let mut stack = Vec::<Frame>::new();
        stack.push(Frame {permutation: Permutation::identity(self.size()), next_basis: 0});

        // Tree-like traversal, while tracking rotations applied to get new rotations and pruning
        // if rotations have been seen before
        while stack.first().unwrap().next_basis <= max_basis_idx {
            let latest = stack.last().unwrap();
            let next_rotation = &self.rotation_basis[latest.next_basis];
            let generated = latest.permutation.compose(&next_rotation).unwrap();

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

    pub fn is_rotation(&self, a: &Permutation, b: &Permutation, rotations: &HashSet<Permutation>) -> bool {
        rotations.iter().any(|r| a.compose(&r).expect("Passed bad rotations") == *b)
    }
}

lazy_static! {
    pub static ref LINE: Shape = Shape {
        name: Name::Line,
        coordinates: Matrix3N::from_column_slice(&[
             1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0
        ]),
        rotation_basis: vec![Permutation {sigma: vec![1, 0]}],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref BENT: Shape = Shape {
        name: Name::Bent,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.292372, 0.956305, 0.0
        ]),
        rotation_basis: vec![Permutation {sigma: vec![1, 0]}],
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
            Permutation {sigma: vec![1, 2, 0]},
            Permutation {sigma: vec![0, 2, 1]}
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
        rotation_basis: vec![Permutation {sigma: vec![2, 0, 1]}],
        tetrahedra: vec![[ORIGIN, 0, 1, 2]],
        mirror: Some(Permutation {sigma: vec![0, 2, 1]})
    };

    pub static ref TSHAPE: Shape = Shape {
        name: Name::T,
        coordinates: Matrix3N::from_column_slice(&[
            -1.0, -0.0, -0.0,
            0.0, 1.0, 0.0,
            1.0, 0.0, 0.0,
        ]),
        rotation_basis: vec![Permutation {sigma: vec![2, 1, 0]}],
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
            Permutation {sigma: vec![0, 3, 1, 2]},
            Permutation {sigma: vec![2, 1, 3, 0]},
            Permutation {sigma: vec![3, 0, 2, 1]},
            Permutation {sigma: vec![1, 2, 0, 3]}
        ],
        tetrahedra: vec![[0, 1, 2, 3]],
        mirror: Some(Permutation {sigma: vec![0, 2, 1, 3]})
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
            Permutation {sigma: vec![3, 0, 1, 2]},
            Permutation {sigma: vec![1, 0, 3, 2]},
            Permutation {sigma: vec![3, 2, 1, 0]},
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
        rotation_basis: vec![Permutation {sigma: vec![3, 2, 1, 0]}],
        tetrahedra: vec![[0, ORIGIN, 1, 2], [ORIGIN, 3, 1, 2]],
        mirror: Some(Permutation {sigma: vec![0, 2, 1, 3]})
    };

    pub static ref TRIGONALPYRAMID: Shape = Shape {
        name: Name::TrigonalPyramid,
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.5, 0.866025, 0.0,
            -0.5, -0.866025, 0.0,
            0.0, 0.0, 1.0
        ]),
        rotation_basis: vec![Permutation {sigma: vec![2, 0, 1, 3]}],
        tetrahedra: vec![[0, 1, 3, 2]],
        mirror: Some(Permutation {sigma: vec![0, 2, 1, 3]})
    };

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT, &EQUILATERAL_TRIANGLE, &VACANT_TETRAHEDRON, &TSHAPE, &TETRAHEDRON, &SQUARE, &SEESAW, &TRIGONALPYRAMID];
}

pub fn shape_from_name(name: Name) -> &'static Shape {
    let shape = SHAPES[name as usize];
    assert_eq!(shape.name, name);
    return shape;
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

fn quaternion_pair_contribution<'a>(stator_col: &na::MatrixSlice3x1<'a, f64>, rotor_col: &na::MatrixSlice3x1<'a, f64>) -> Matrix4 {
    let mut a = Matrix4::zeros();

    let forward_difference = (rotor_col - stator_col).transpose();
    a.fixed_slice_mut::<1, 3>(0, 1).copy_from(&forward_difference);

    let backward_difference = stator_col - rotor_col;
    a.fixed_slice_mut::<3, 1>(1, 0).copy_from(&backward_difference);

    let mut block = Matrix3::zeros();
    let sum = stator_col + rotor_col;
    for (col, mut block_col) in Matrix3::identity().column_iter().zip(block.column_iter_mut()) {
        block_col.copy_from(&col.cross(&sum));
    }
    a.fixed_slice_mut::<3, 3>(1, 1).copy_from(&block);

    a.transpose() * a
}

fn quaternion_decomposition(mat: Matrix4) -> Quaternion {
    let decomposition = na::SymmetricEigen::new(mat);
    let min_eigenvalue_index = decomposition.eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Encountered NaN"))
        .map(|(index, _)| index)
        .expect("No eigenvalues present");

    let q = decomposition.eigenvectors.column(min_eigenvalue_index);
    let quaternion = na::Quaternion::new(q[0], q[1], q[2], q[3]);
    na::UnitQuaternion::from_quaternion(quaternion)
}

/// Find a quaternion that best transforms the stator into the rotor
///
/// Postcondition is rotor = quat * stator and stator = quat.inverse() * rotor
pub fn fit_quaternion(stator: &Matrix3N, rotor: &Matrix3N) -> Quaternion {
    // Ensure centroids are removed from matrices
    assert!(stator.column_mean().norm_squared() < 1e-8);
    assert!(rotor.column_mean().norm_squared() < 1e-8);

    let mut a = Matrix4::zeros();
    // Generate decomposable matrix per atom and add them
    for (rotor_col, stator_col) in rotor.column_iter().zip(stator.column_iter()) {
        a += quaternion_pair_contribution(&stator_col, &rotor_col);
    }

    quaternion_decomposition(a)
}

pub fn fit_quaternion_with_map(stator: &Matrix3N, rotor: &Matrix3N, vertex_map: &HashMap<usize, usize>) -> Quaternion {
    // Ensure centroids are removed from matrices
    assert!(stator.column_mean().norm_squared() < 1e-8);
    assert!(rotor.column_mean().norm_squared() < 1e-8);
    
    let mut a = Matrix4::zeros();
    for (stator_i, rotor_i) in vertex_map {
        let stator_col = stator.column(*stator_i);
        let rotor_col = rotor.column(*rotor_i);
        a += quaternion_pair_contribution(&stator_col, &rotor_col);
    }

    quaternion_decomposition(a)
}

pub fn apply_permutation(x: &Matrix3N, p: &Permutation) -> Matrix3N {
    assert_eq!(x.ncols(), p.sigma.len());
    let inverse = p.inverse();
    Matrix3N::from_fn(x.ncols(), |i, j| x[(i, inverse[j] as usize)])
}

mod scaling {
    use crate::shapes::Matrix3N;

    use argmin::prelude::*;
    use argmin::solver::goldensectionsearch::GoldenSectionSearch;

    struct CoordinateScalingProblem<'a> {
        pub cloud: &'a Matrix3N,
        pub rotated_shape: &'a Matrix3N
    }

    impl ArgminOp for CoordinateScalingProblem<'_> {
        type Param = f64;
        type Output = f64;
        type Hessian = ();
        type Jacobian = ();
        type Float = f64;

        fn apply(&self, p: &Self::Param) -> Result<Self::Output, Error> {
            let rmsd = (self.cloud.clone() - self.rotated_shape.scale(*p))
                .column_iter()
                .map(|col| col.norm_squared())
                .sum();

            Ok(rmsd)
        }
    }

    pub fn minimize(cloud: &Matrix3N, shape: &Matrix3N) -> f64 {
        let phi_sectioning = GoldenSectionSearch::new(0.5, 1.1);
        let scaling_problem = CoordinateScalingProblem {
            cloud: &cloud,
            rotated_shape: &shape 
        };
        let result = Executor::new(scaling_problem, phi_sectioning, 1.0)
            .max_iters(100)
            .run()
            .unwrap();

        let result_rmsd = result.state.best_cost;
        let normalization: f64 = cloud.column_iter().map(|v| v.norm_squared()).sum();
        100.0 * result_rmsd / normalization
    }
}

#[derive(Error, Debug, PartialEq)]
pub enum SimilarityError {
    #[error("Number of positions does not match shape size")]
    PositionNumberMismatch
}

pub fn polyhedron_similarity(x: &Matrix3N, s: Name) -> Result<(Permutation, f64), SimilarityError> {
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let mut permutation = Permutation::identity(n);
    let cloud = unit_sphere_normalize(x.clone());
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);

    let evaluate_permutation = |p: Permutation| -> (Permutation, f64, Quaternion) {
        let permuted_shape = apply_permutation(&shape_coordinates, &p);
        let quaternion = fit_quaternion(&cloud, &permuted_shape);
        let permuted_shape = quaternion.inverse().to_rotation_matrix() * permuted_shape;

        let rmsd = (cloud.clone() - permuted_shape)
            .column_iter()
            .map(|col| col.norm_squared())
            .sum();

        (p, rmsd, quaternion)
    };

    let (best_permutation, _, best_quaternion) = permutation
        .iter()
        .map(evaluate_permutation)
        .min_by(|(_, rmsd_a, _), (_, rmsd_b, _)| rmsd_a.partial_cmp(rmsd_b).expect("NaN in RMSDs"))
        .expect("Not a single permutation available to try");

    let permuted_shape = apply_permutation(&shape_coordinates, &best_permutation);
    let rotated_shape = best_quaternion.inverse().to_rotation_matrix() * permuted_shape;

    let csm = scaling::minimize(&cloud, &rotated_shape);
    Ok((best_permutation, csm))
}

pub fn polyhedron_similarity_shortcut(x: &Matrix3N, s: Name) -> Result<(Permutation, f64), SimilarityError> {
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size() + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let n_prematch = 5;

    if n < n_prematch {
        return polyhedron_similarity(x, s);
    }

    let cloud = unit_sphere_normalize(x.clone());
    // TODO check if the centroid is last, e.g. by ensuring it is the shortest vector after normalization

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);

    type PartialPermutation = HashMap<usize, usize>;

    struct Narrowing {
        penalty: f64,
        mapping: PartialPermutation,
    }

    let narrow = (0..n).permutations(n_prematch).fold(
        Narrowing {penalty: f64::MAX, mapping: PartialPermutation::new()},
        |best_narrow, vertices| -> Narrowing {
            let mut partial_map = PartialPermutation::with_capacity(n);
            vertices.iter().enumerate().for_each(|(k, v)| { partial_map.insert(k, *v); });
            let partial_quaternion = fit_quaternion_with_map(&cloud, &shape_coordinates, &partial_map); 
            let rotated_shape = partial_quaternion.inverse().to_rotation_matrix() * shape_coordinates.clone();

            let penalty: f64 = vertices.iter()
                .enumerate()
                .map(|(i, j)| (cloud.column(i) - rotated_shape.column(*j)).norm_squared())
                .sum();

            if penalty > best_narrow.penalty {
                return best_narrow;
            }

            let left_free: Vec<usize> = (n_prematch..n).collect();
            let right_free: Vec<usize> = (0..n).filter(|i| !vertices.contains(i)).collect();
            assert_eq!(left_free.len(), right_free.len());
            let v = left_free.len();

            let costs = na::DMatrix::from_fn(v, v, |i, j| (cloud.column(i) - rotated_shape.column(j)).norm_squared());
            // Brute-force the costs matrix permutation
            let (_, sub_permutation) = Permutation::identity(v).iter()
                .map(|permutation| ((0..v).map(|i| costs[(i, permutation[i] as usize)]).sum(), permutation) )
                .min_by(|(a, _): &(f64, Permutation), (b, _): &(f64, Permutation)| a.partial_cmp(b).expect("Encountered NaNs"))
                                                                .expect("At least one permutation present");

            // Fuse permutation and best subpermutation
            sub_permutation.sigma.iter().enumerate().for_each(|(i, j)| { partial_map.insert(left_free[i], right_free[*j as usize]); });

            // Make a clean quaternion fit with the full mapping
            let full_fit_quat = fit_quaternion_with_map(&cloud, &shape_coordinates, &partial_map);
            let full_fit_rotated = full_fit_quat.inverse().to_rotation_matrix() * shape_coordinates.clone();
            let full_penalty = partial_map.iter().map(|(i, j)| (cloud.column(*i) - full_fit_rotated.column(*j)).norm_squared()).sum();

            Narrowing {penalty: full_penalty, mapping: partial_map}
        }
    );

    /* Given the best permutation for the rotational fit, we still have to
     * minimize over the isotropic scaling factor. It is cheaper to reorder the
     * positions once here for the minimization so that memory access is in-order
     * during the repeated scaling minimization function call.
     */
    let mut best_permutation = Permutation::identity(n);
    narrow.mapping.iter().for_each(|(i, j)| { best_permutation.sigma[*i] = *j as u8; });

    let permuted_shape = apply_permutation(&shape_coordinates, &best_permutation);
    let quaternion = fit_quaternion(&cloud, &permuted_shape);
    let rotated_shape = quaternion.inverse().to_rotation_matrix() * permuted_shape;

    let csm = scaling::minimize(&cloud, &rotated_shape);
    Ok((best_permutation, csm))
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
        Permutation::from_index(n, random_discrete(Permutation::count(n)))
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
    fn test_quaternion_fit() {
        let stator = random_cloud(6);
        let true_quaternion = random_rotation();

        let rotor = true_quaternion.to_rotation_matrix() * stator.clone();
        let fitted_quaternion = fit_quaternion(&stator, &rotor);
        approx::assert_relative_eq!(true_quaternion, fitted_quaternion, epsilon = 1e-6);

        approx::assert_relative_eq!(rotor, fitted_quaternion.to_rotation_matrix() * stator.clone(), epsilon=1e-6);
        approx::assert_relative_eq!(stator, fitted_quaternion.inverse().to_rotation_matrix() * rotor, epsilon=1e-6);
    }

    #[test]
    fn test_apply_permutation() {
        let n = 6;
        // upper bound is 6! = 720, u8::MAX is 255, which will do
        let permutation_index = rand::random::<u8>() as usize; 

        let stator = random_cloud(n);
        let permutation = Permutation::from_index(n, permutation_index);
        let permuted = apply_permutation(&stator, &permutation);

        for i in 0..n {
            assert_eq!(stator.column(i), permuted.column(permutation[i] as usize));
        }
    }

    #[test]
    fn test_normalization() {
        let cloud = random_cloud(6);

        // No centroid
        let centroid_norm = cloud.column_mean().norm();
        assert!(centroid_norm < 1e-6);

        // Longest vector is a unit vector
        let longest_norm = cloud.column_iter().map(|v| v.norm()).max_by(|a, b| a.partial_cmp(b).expect("Encountered NaNs")).unwrap();
        approx::assert_relative_eq!(longest_norm, 1.0);
    }

    #[test]
    fn shape_self_similarity() {
        let fns = [polyhedron_similarity, polyhedron_similarity_shortcut];
        let fn_names = ["similarity", "similarity_shortcut"];

        for shape in SHAPES.iter() {
            let shape_rotations = shape.generate_rotations();

            let cloud = shape.coordinates.clone().insert_column(shape.size(), 0.0);
            let random_rotation = random_rotation().to_rotation_matrix();
            let rotated_cloud = unit_sphere_normalize(random_rotation * cloud);
            let random_permutation = random_permutation(rotated_cloud.ncols());
            let permuted_cloud = apply_permutation(&rotated_cloud, &random_permutation);

            let results: Vec<(Permutation, f64)> = fns.iter().map(|f| f(&permuted_cloud, shape.name).expect("Fine")).collect();

            for (fn_name, (permutation, similarity)) in fn_names.iter().zip(&results) {
                println!("Algorithm {} achieved similarity {} of {} with permutation {}", fn_name, similarity, shape.name.repr(), permutation);
                assert!(*similarity < 1e-6);
                // Centroid must be last
                assert_eq!(*permutation.sigma.last().unwrap(), shape.size() as u8);
            }

            let permutation_results : Vec<Permutation> = results.iter().map(|(p, _)| { let mut q = p.clone(); q.sigma.pop(); q }).collect();
            let mutual_rotations = permutation_results.iter().tuple_windows().all(|(p, q)| shape.is_rotation(p, q, &shape_rotations));
            assert!(mutual_rotations);
        }
    }

    // for v in cloud.column_iter_mut() {
    //     v += Vector3::new_random().normalize().scale(0.1);
    // }
}
