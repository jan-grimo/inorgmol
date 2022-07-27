extern crate nalgebra as na;
use na::{Matrix, U3, Dynamic, VecStorage, Vector3};

pub enum Name {
    Line,
    Bent
}

pub type Matrix3N = Matrix<f64, U3, Dynamic, VecStorage<f64, U3, Dynamic>>;

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

pub fn centroid(x: &Matrix3N) -> Vector3<f64> {
    let mut center = Vector3::zeros();
    for c in x.column_iter() {
        center += c;
    }
    center /= x.ncols() as f64;
    center
}

pub fn normalize(x: &mut Matrix3N) {
    // Remove centroid
    let center = centroid(x);
    for mut c in x.column_iter_mut() {
        c -= center;
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
}

pub fn polyhedron_similarity(x: &Matrix3N, s: Name) -> f64 {
    let n = x.ncols();
    let mut cloud = x.clone();
    normalize(&mut cloud);

    let mut permutation = Permutation::identity(n);

    let shape_coordinates = &SHAPES[s as usize].coordinates;

    0.1
}
