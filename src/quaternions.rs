extern crate nalgebra as na;

pub type Matrix3N = na::Matrix3xX<f64>;
pub type Matrix3 = na::Matrix3<f64>;
pub type Quaternion = na::UnitQuaternion<f64>;
type Matrix4 = na::Matrix4<f64>;

use std::collections::HashMap;

use derive_more::From;

pub fn random_rotation() -> Quaternion {
    let random_axis = na::Unit::new_normalize(na::Vector3::new_random());
    let random_angle = rand::random::<f64>() * std::f64::consts::PI;
    Quaternion::from_axis_angle(&random_axis, random_angle)
}

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

pub struct Fit {
    pub quaternion: Quaternion,
    pub msd: f64
}

impl Fit {
    pub fn rotate_stator(&self, stator: &Matrix3N) -> Matrix3N {
        self.quaternion.to_rotation_matrix() * stator
    }
    pub fn rotate_rotor(&self, rotor: &Matrix3N) -> Matrix3N {
        self.quaternion.inverse().to_rotation_matrix() * rotor
    }
}

pub fn quaternion_decomposition(mat: Matrix4) -> Fit {
    let decomposition = na::SymmetricEigen::new(mat);
    // Eigenvalues are unsorted here, we seek the minimum value
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

/// Find a quaternion that best transforms the stator into the rotor
///
/// Postconditions 
/// - rotor = quat * stator and stator = quat.inverse() * rotor,
///   see rotate_rotor and rotate_stator fns
/// - The resulting quaternion is a proper rotation (no inversions)
pub fn fit(stator: &Matrix3N, rotor: &Matrix3N) -> Fit {
    // Ensure centroids are removed from matrices
    assert!(stator.column_mean().norm_squared() < 1e-8);
    assert!(rotor.column_mean().norm_squared() < 1e-8);

    let mut a = Matrix4::zeros();
    // Generate decomposable matrix per coordinate pair and add them
    for (rotor_col, stator_col) in rotor.column_iter().zip(stator.column_iter()) {
        a += quaternion_pair_contribution(&stator_col, &rotor_col);
    }

    quaternion_decomposition(a)
}

pub fn fit_with_map(stator: &Matrix3N, rotor: &Matrix3N, vertex_map: &HashMap<usize, usize>) -> Fit {
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

#[derive(Debug, Clone, From)]
pub struct Stator {
    pub matrix: Matrix3N
}

#[derive(Debug, Clone, From)]
pub struct Rotor {
    pub matrix: Matrix3N
}

impl Stator {
    pub fn fit(&self, rotor: &Rotor) -> Fit {
        crate::quaternions::fit(&self.matrix, &rotor.matrix)
    }

    pub fn fit_with_map(&self, rotor: &Rotor, vertex_map: &HashMap<usize, usize>) -> Fit {
        crate::quaternions::fit_with_map(&self.matrix, &rotor.matrix, vertex_map)
    }

    pub fn rotate(self, fit: &Fit) -> Matrix3N {
        fit.quaternion.to_rotation_matrix() * self.matrix
    }
}

impl Rotor {
    pub fn rotate(self, fit: &Fit) -> Matrix3N {
        fit.quaternion.inverse().to_rotation_matrix() * self.matrix
    }
}

#[cfg(test)]
mod tests {
    use crate::quaternions::*;
    use crate::shapes::similarity::{unit_sphere_normalize, apply_permutation};
    use crate::permutation::Permutation;

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
            let stator = Stator::from(random_cloud(v));
            let quaternion = random_rotation();
            let rotor = Rotor::from(quaternion.to_rotation_matrix() * stator.matrix.clone());

            Case {stator, rotor, quaternion}
        }
    }

    #[test]
    fn basics() {
        let case = Case::new(6);

        let fit = case.stator.fit(&case.rotor);
        approx::assert_relative_eq!(case.quaternion, fit.quaternion, epsilon = 1e-6);

        // Test postconditions
        approx::assert_relative_eq!(case.rotor.matrix, fit.rotate_stator(&case.stator.matrix), epsilon=1e-6);
        approx::assert_relative_eq!(case.rotor.matrix, case.stator.clone().rotate(&fit), epsilon=1e-6);
        approx::assert_relative_eq!(case.stator.matrix, fit.rotate_rotor(&case.rotor.matrix), epsilon=1e-6);
        approx::assert_relative_eq!(case.stator.matrix, case.rotor.rotate(&fit), epsilon=1e-6);
    }

    #[test]
    fn inexact_fit_msd_accurate() {
        let v = 6;
        let case = Case::new(v);

        // Inexact fit msd from eigenvalue is accurate
        let distorted_rotor = Rotor::from(case.rotor.matrix + 0.01 * random_cloud(v));
        let distorted_fit = case.stator.fit(&distorted_rotor);
        let rotated_rotor = distorted_fit.quaternion.inverse().to_rotation_matrix() * distorted_rotor.matrix;
        let msd: f64 = (case.stator.matrix - rotated_rotor)
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
        let permuted_rotor = Rotor::from(apply_permutation(&case.rotor.matrix, &permutation));
        let partial_permutation = {
            let mut p = HashMap::new();
            for (i, j) in permutation.iter_pairs().take(3) {
                p.insert(i, *j as usize);
            }
            p
        };
        let partial_fit = case.stator.fit_with_map(&permuted_rotor, &partial_permutation);

        approx::assert_relative_eq!(partial_fit.msd, 0.0, epsilon = 1e-6);
    }
}
