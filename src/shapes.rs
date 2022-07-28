extern crate nalgebra as na;

pub type Matrix3N = na::Matrix3xX<f64>;
pub type Matrix3 = na::Matrix3<f64>;
pub type Quaternion = na::UnitQuaternion<f64>;

extern crate thiserror;
use thiserror::Error;

pub enum Name {
    Line,
    Bent
}

use crate::permutation;
pub type Permutation = permutation::Permutation;

pub struct Shape {
    pub name: Name,
    pub size: u8,
    pub repr: &'static str,
    pub coordinates: Matrix3N,
    pub rotations: Vec<Permutation>,
    pub tetrahedra: Vec<[u8; 4]>,
    pub mirror: Option<Permutation>
}

lazy_static! {
    pub static ref LINE: Shape = Shape {
        name: Name::Line,
        size: 2,
        repr: "line",
        coordinates: Matrix3N::from_column_slice(&[
             1.0, 0.0, 0.0,
            -1.0, 0.0, 0.0
        ]),
        rotations: vec![Permutation {sigma: vec![1, 0]}],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref BENT: Shape = Shape {
        name: Name::Bent,
        size: 2,
        repr: "bent",
        coordinates: Matrix3N::from_column_slice(&[
            1.0, 0.0, 0.0,
            -0.292372, 0.956305, 0.
        ]),
        rotations: vec![Permutation {sigma: vec![1, 0]}],
        tetrahedra: vec![],
        mirror: None
    };

    pub static ref SHAPES: Vec<&'static Shape> = vec![&LINE, &BENT];
}

pub fn shape_from_name(name: Name) -> &'static Shape {
    return SHAPES[name as usize];
}

pub fn unit_sphere_normalize(mut x: Matrix3N) -> Matrix3N {
    // Remove centroid
    let centroid = x.column_mean();
    for mut c in x.column_iter_mut() {
        c -= centroid;
    }

    // Find the longest vector
    let longest_norm = x.column_iter().try_fold(
        0.0,
        |acc, x| {
            let norm = x.norm();
            let cmp = norm.partial_cmp(&acc)?;
            let max = if let std::cmp::Ordering::Greater = cmp {
                norm
            } else {
                acc
            };
            Some(max)
        }
    ).expect("Finding longest vector in positions encountered NaNs");

    // Rescale all distances so that the longest is a unit vector
    for mut c in x.column_iter_mut() {
        c /= longest_norm;
    }

    x
}

/// Find a rotation that transforms the rotor into the stator
pub fn quaternion_fit(stator: &Matrix3N, rotor: &Matrix3N) -> Quaternion {
    // Ensure centroids are removed from matrices
    assert!(stator.column_mean().norm_squared() < 1e-8);
    assert!(rotor.column_mean().norm_squared() < 1e-8);

    type Matrix4 = na::Matrix4<f64>;

    let mut b = Matrix4::zeros();
    // Generate decomposable matrix per atom and add them
    for (rotor_col, stator_col) in rotor.column_iter().zip(stator.column_iter()) {
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

        b += a.transpose() * a;
    }

    let decomposition = na::SymmetricEigen::new(b);
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

#[derive(Error, Debug)]
pub enum SimilarityError {
    #[error("Number of positions does not match shape size")]
    PositionNumberMismatch
}

pub fn polyhedron_similarity(x: &Matrix3N, s: Name) -> Result<f64, SimilarityError> {
    let shape = shape_from_name(s);
    let n = x.ncols();
    if n != shape.size as usize + 1 {
        return Err(SimilarityError::PositionNumberMismatch);
    }

    let mut permutation = Permutation::identity(n);
    let cloud = unit_sphere_normalize(x.clone());
    let shape_coordinates = unit_sphere_normalize(shape.coordinates.clone().insert_column(n, 0.0));

    Ok(0.1)
}

#[cfg(test)]
mod tests {
    use crate::shapes::{unit_sphere_normalize, quaternion_fit, Matrix3N, Quaternion};
    use nalgebra::Vector3;

    #[test]
    fn test_quaternion_fit() {
        let stator = unit_sphere_normalize(Matrix3N::new_random(6));
        let true_quaternion = Quaternion::from_axis_angle(&Vector3::y_axis(), std::f64::consts::FRAC_PI_2);
        let rotor = true_quaternion.to_rotation_matrix().matrix() * stator.clone();
        let fitted_quaternion = quaternion_fit(&stator, &rotor);
        approx::assert_relative_eq!(true_quaternion, fitted_quaternion, epsilon = 1e-6);
    }
}
