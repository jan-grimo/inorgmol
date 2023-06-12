use molassembler::shapes::similarity::{unit_sphere_normalize, apply_permutation, Similarity, SimilarityError};
use molassembler::quaternions::Matrix3N;
use molassembler::permutation::Permutation;
use molassembler::strong::bijection::Bijection;
use molassembler::quaternions::random_rotation;
use molassembler::shapes::*;

use std::collections::{HashMap, HashSet};
use itertools::Itertools;

// NOTES
// - sample output: 
//
//   4 shapes of size 6, 100 repetitions
//   prematch(5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//   prematch(4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
//   prematch(3): [0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2]
//   
//   3 shapes of size 7, 100 repetitions
//   prematch(5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//   prematch(4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
//   prematch(3): [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
//   
//   4 shapes of size 8, 100 repetitions
//   prematch(5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
//   prematch(4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5]
//   prematch(3): [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 10]
//
//   3 shapes of size 9, 100 repetitions
//   prematch(5): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
//   prematch(4): [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3]
//   prematch(3): [0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 10]
//   

// TODO 
// - Review implementations and find a failure criterion for falling back onto
//   higher prematch count implementations

struct Case {
    shape_name: Name,
    cloud: Matrix3N,
    expected_bijection: Bijection<Vertex, Column>
}

impl Case {
    fn distort(mut mat: Matrix3N, norm: f64) -> Matrix3N {
        for mut v in mat.column_iter_mut() {
            v += norm * nalgebra::Vector3::new_random().normalize();
        }

        mat
    }

    fn rotate(mat: Matrix3N) -> Matrix3N {
        random_rotation().to_rotation_matrix() * mat
    }

    fn permute(mat: Matrix3N) -> (Matrix3N, Bijection<Vertex, Column>) {
        let bijection: Bijection<Vertex, Column> = {
            let mut p = Permutation::new_random(mat.ncols() - 1);
            p.sigma.push(p.set_size());
            Bijection::new(p)
        };

        let permuted = apply_permutation(&mat, &bijection.permutation);

        (permuted, bijection)
    }

    fn pop_centroid(mut bijection: Bijection<Vertex, Column>) -> Bijection<Vertex, Column> {
        let _ = bijection.permutation.sigma.pop().expect("No zero-length bijections");
        bijection
    }

    pub fn new(shape: &Shape, distortion_norm: f64) -> Case {
        let coords = shape.coordinates.clone().insert_column(shape.num_vertices(), 0.0);
        let distorted = Self::distort(Self::rotate(coords), distortion_norm);
        // The bijection with which the distorted version is generated is not
        // necessarily useful for comparison since it's possible the distorted
        // version could have a better bijection!
        let (distorted, bijection) = Self::permute(distorted);
        let cloud = unit_sphere_normalize(distorted);

        Case {shape_name: shape.name, cloud, expected_bijection: bijection}
    }

    pub fn pass(&self, f: &dyn Fn(Matrix3N, &Shape) -> Result<Similarity, SimilarityError>, rotations: &HashSet<molassembler::shapes::Rotation>) -> bool {
        let shape = shape_from_name(self.shape_name);
        let f_similarity = f(self.cloud.clone(), shape).expect("similarity fn doesn't panic");

        let expected_bijection = Self::pop_centroid(self.expected_bijection.clone());
        let found_bijection = Self::pop_centroid(f_similarity.bijection);
        let is_rotation_of_expected = Shape::is_rotation(&expected_bijection, &found_bijection, rotations);

        if !is_rotation_of_expected {
            // It's possible a strongly distorted version actually has a better
            // fitting bijection than the one it was made with, so check
            // with a reference method

            let similarity = similarity::polyhedron_reference(self.cloud.clone(), shape).expect("Reference doesn't fail");
            let ref_bijection = Self::pop_centroid(similarity.bijection);
            let is_rotation = Shape::is_rotation(&ref_bijection, &found_bijection, rotations);

            let csm_close = (similarity.csm - f_similarity.csm).abs() < 1e-3;

            csm_close && is_rotation
        } else {
            true
        }
    }
}

#[derive(Clone)]
struct Statistics<'a> {
    f: &'a dyn Fn(Matrix3N, &Shape) -> Result<Similarity, SimilarityError>,
    name: String,
    distortion_failures: HashMap<usize, usize>
}

impl<'a> Statistics<'a> {
    pub fn new(f: &'a dyn Fn(Matrix3N, &Shape) -> Result<Similarity, SimilarityError>, name: impl Into<String>) -> Statistics<'a> {
        Statistics {f, name: name.into(), distortion_failures: HashMap::new()}
    }
}

fn flatten(map: &HashMap<usize, usize>) -> Vec<usize> {
    Vec::from_iter((0..=10).map(|i| map[&i]))
}

fn main() {
    for (shape_size, shapes) in &SHAPES.iter().group_by(|s| s.num_vertices()) {
        if !(6..10).contains(&shape_size) {
            continue;
        }

        const REPETITIONS: usize = 100;

        let mut stats = [
            // Statistics::new(&similarity::polyhedron_reference, "reference"),
            Statistics::new(&similarity::polyhedron_base::<5, true, true>, "prematch(5)"),
            Statistics::new(&similarity::polyhedron_base::<4, true, true>, "prematch(4)"),
            Statistics::new(&similarity::polyhedron_base::<3, true, true>, "prematch(3)"),
        ].to_vec();


        let mut shape_count = 0;
        for shape in shapes {
            let rotations = shape.generate_rotations();

            for i in 0..=10 {
                let distortion_norm = 0.1 * i as f64;

                for stat in stats.iter_mut() {
                    let fail_count = (0..REPETITIONS)
                        .map(|_| Case::new(shape, distortion_norm).pass(stat.f, &rotations))
                        .filter(|s| !*s)
                        .count();

                    if let Some(existing_failures) = stat.distortion_failures.get_mut(&i) {
                        *existing_failures += fail_count;
                    } else {
                        stat.distortion_failures.insert(i, fail_count);
                    }
                }
            }

            shape_count += 1;
        }

        println!("{} shapes of size {}, {} repetitions", shape_count, shape_size, REPETITIONS);
        for stat in stats {
            println!("{}: {:?}", stat.name, flatten(&stat.distortion_failures));
        }
        println!();
    }
}
