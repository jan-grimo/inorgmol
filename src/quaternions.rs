extern crate nalgebra as na;

/// Three dimensional matrix type
pub type Matrix3N = na::Matrix3xX<f64>;
/// Square three dimensional matrix type
pub type Matrix3 = na::Matrix3<f64>;
/// Quaternion type
pub type Quaternion = na::UnitQuaternion<f64>;
type Matrix4 = na::Matrix4<f64>;

use std::collections::HashMap;

use derive_more::From;

/// Generate a random rotation quaternion
pub fn random_rotation() -> Quaternion {
    let random_axis = na::Unit::new_normalize(na::Vector3::new_random());
    let random_angle = rand::random::<f64>() * std::f64::consts::PI;
    Quaternion::from_axis_angle(&random_axis, random_angle)
}

/// Calculate quaternion matrix contribution for a pair of points
pub fn quaternion_pair_contribution<'a>(stator_col: &na::MatrixView3x1<'a, f64>, rotor_col: &na::MatrixView3x1<'a, f64>) -> Matrix4 {
    let mut a = Matrix4::zeros();

    let forward_difference = (rotor_col - stator_col).transpose();
    a.fixed_view_mut::<1, 3>(0, 1).copy_from(&forward_difference);

    let backward_difference = stator_col - rotor_col;
    a.fixed_view_mut::<3, 1>(1, 0).copy_from(&backward_difference);

    let mut block = Matrix3::zeros();
    let sum = stator_col + rotor_col;
    for (col, mut block_col) in Matrix3::identity().column_iter().zip(block.column_iter_mut()) {
        block_col.copy_from(&col.cross(&sum));
    }
    a.fixed_view_mut::<3, 3>(1, 1).copy_from(&block);

    a.transpose() * a
}

/// Fitting result between two matrices
pub struct Fit {
    /// Quaternion transforming the stator into the rotor
    pub quaternion: Quaternion,
    /// Mean square deviation
    pub msd: f64
}

impl Fit {
    /// Rotate the stator to align with the rotor
    ///
    /// See [`Stator`] for a strongly-typed solution to avoiding passing the wrong matrix
    pub fn rotate_stator(&self, stator: &Matrix3N) -> Matrix3N {
        self.quaternion.to_rotation_matrix() * stator
    }

    /// Rotate the rotor to align with the stator
    ///
    /// See [`Stator`] for a strongly-typed solution to avoid passing the wrong matrix
    pub fn rotate_rotor(&self, rotor: &Matrix3N) -> Matrix3N {
        self.quaternion.inverse().to_rotation_matrix() * rotor
    }
}

/// Decompose the sum of pair contributions into a fit
///
/// See [`quaternion_pair_contribution`] to compose `mat`.
pub fn quaternion_decomposition(mat: Matrix4) -> Fit {
    let decomposition = na::SymmetricEigen::new(mat);
    // Eigenvalues are unsorted here, we seek the minimum value
    //
    // NOTE: Best inverted fit uses eigenvector of largest eigenvector l_3 
    // with msd of 0.5 * (l_0 + l_1 + l_2 - l_3), so can check if inversion
    // better quite easily if desired
    let (min_eigenvalue_index, msd) = decomposition.eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("Encountered NaN"))
        .expect("No eigenvalues present");

    let q = decomposition.eigenvectors.column(min_eigenvalue_index);
    let pre_quat = na::Quaternion::new(q[0], q[1], q[2], q[3]);

    Fit {
        quaternion: na::UnitQuaternion::from_quaternion(pre_quat),
        msd: *msd
    }
}

/// Test whether a matrix has had its centroid removed
pub fn centroidless(mat: &Matrix3N) -> bool {
    mat.column_mean().norm_squared() < 1e-8
}

/// Find a quaternion that best transforms the stator into the rotor
pub fn fit(stator: &Matrix3N, rotor: &Matrix3N) -> Fit {
    let mut a = Matrix4::zeros();
    // Generate decomposable matrix per coordinate pair and add them
    for (rotor_col, stator_col) in rotor.column_iter().zip(stator.column_iter()) {
        a += quaternion_pair_contribution(&stator_col, &rotor_col);
    }

    quaternion_decomposition(a)
}

/// Fit a quaternion onto two point clouds with an index mapping between them
///
/// Only adds pair contributions of vertex pairs in `vertex_map`. Unmapped vertices do not
/// contribute to the quaternion fit.
///
/// Requires that both matrices have their offset centroid removed.
pub fn fit_with_map(stator: &Matrix3N, rotor: &Matrix3N, vertex_map: &HashMap<usize, usize>) -> Fit {
    let mut a = Matrix4::zeros();
    for (stator_i, rotor_i) in vertex_map {
        let stator_col = stator.column(*stator_i);
        let rotor_col = rotor.column(*rotor_i);
        a += quaternion_pair_contribution(&stator_col, &rotor_col);
    }

    quaternion_decomposition(a)
}

fn remove_offset(mut vertices: Matrix3N) -> Matrix3N {
    let centroid = vertices.column_mean();
    if centroid.norm_squared() > 1e-3 {
        for mut v in vertices.column_iter_mut() {
            v -= centroid;
        }
    }

    vertices
}

/// Fit a quaternion between two point clouds
///
/// Removes any offset centroid that is present
pub fn fit_remove_offset(stator: Matrix3N, rotor: Matrix3N) -> Fit {
    fit(&remove_offset(stator), &remove_offset(rotor))
}

/// New type stator matrix
#[derive(Debug, Clone, From)]
pub struct Stator(pub Matrix3N);

/// New type rotor matrix
#[derive(Debug, Clone, From)]
pub struct Rotor(pub Matrix3N);

impl Stator {
    /// Find a quaternion fit between this stator and a rotor
    ///
    /// See [`fit`]
    pub fn fit(&self, rotor: &Rotor) -> Fit {
        crate::quaternions::fit(&self.0, &rotor.0)
    }

    /// Find a quaternion fit between this stator and a rotor with an index mapping
    ///
    /// See [`fit_with_map`]
    pub fn fit_with_map(&self, rotor: &Rotor, vertex_map: &HashMap<usize, usize>) -> Fit {
        crate::quaternions::fit_with_map(&self.0, &rotor.0, vertex_map)
    }

    /// Rotate the stator to best fit the rotor
    pub fn rotate(self, fit: &Fit) -> Matrix3N {
        fit.quaternion.to_rotation_matrix() * self.0
    }
}

impl Rotor {
    /// Rotate the rotor to best fit the stator
    pub fn rotate(self, fit: &Fit) -> Matrix3N {
        fit.quaternion.inverse().to_rotation_matrix() * self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::quaternions::*;
    use crate::shapes::similarity::unit_sphere_normalize;
    use crate::permutation::{Permutation, Permutatable};

    fn random_cloud(n: usize) -> Matrix3N {
        unit_sphere_normalize(Matrix3N::new_random(n))
    }

    struct Case {
        stator: Stator,
        rotor: Rotor,
        quaternion: Quaternion
    }

    impl Case {
        fn new(v: usize) -> Case {
            let stator = Stator(random_cloud(v));
            let quaternion = random_rotation();
            let rotor = Rotor(quaternion.to_rotation_matrix() * stator.0.clone());

            Case {stator, rotor, quaternion}
        }
    }

    #[test]
    fn basics() {
        let case = Case::new(6);

        let fit = case.stator.fit(&case.rotor);
        approx::assert_relative_eq!(case.quaternion, fit.quaternion, epsilon = 1e-6);

        // Test postconditions
        approx::assert_relative_eq!(case.rotor.0, fit.rotate_stator(&case.stator.0), epsilon=1e-6);
        approx::assert_relative_eq!(case.rotor.0, case.stator.clone().rotate(&fit), epsilon=1e-6);
        approx::assert_relative_eq!(case.stator.0, fit.rotate_rotor(&case.rotor.0), epsilon=1e-6);
        approx::assert_relative_eq!(case.stator.0, case.rotor.rotate(&fit), epsilon=1e-6);
    }

    #[test]
    fn inexact_fit_msd_accurate() {
        let v = 6;
        let case = Case::new(v);

        // Inexact fit msd from eigenvalue is accurate
        let distorted_rotor = Rotor::from(case.rotor.0 + 0.01 * random_cloud(v));
        let distorted_fit = case.stator.fit(&distorted_rotor);
        let rotated_rotor = distorted_fit.quaternion.inverse().to_rotation_matrix() * distorted_rotor.0;
        let msd: f64 = (case.stator.0 - rotated_rotor)
            .column_iter()
            .map(|col| col.norm_squared())
            .sum();

        approx::assert_relative_eq!(msd, distorted_fit.msd, epsilon = 1e-6);
    }

    #[test]
    fn with_map_zero_msd() {
        let v = 6;
        let case = Case::new(v);
        let permutation = Permutation::new_random(v);
        let permuted_rotor = Rotor::from(case.rotor.0.permute(&permutation).expect("Matching size"));
        let partial_permutation = {
            let mut p = HashMap::new();
            for (i, j) in permutation.iter().take(3) {
                p.insert(i, *j);
            }
            p
        };
        let partial_fit = case.stator.fit_with_map(&permuted_rotor, &partial_permutation);

        approx::assert_relative_eq!(partial_fit.msd, 0.0, epsilon = 1e-6);
    }
}
