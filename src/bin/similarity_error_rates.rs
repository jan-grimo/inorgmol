use molassembler::shapes::similarity::{unit_sphere_normalize, apply_permutation, Similarity, SimilarityError};
use molassembler::quaternions::Matrix3N;
use molassembler::permutation::Permutation;
use molassembler::strong::bijection::Bijection;
use molassembler::quaternions::random_rotation;
use molassembler::shapes::*;

// TODO 
// - What to do with Case instances where the initial bijection isn't even 
//   found by non-skipping brute force?
// - Abstract over shapes

struct Case {
    shape_name: Name,
    cloud: Matrix3N,
    similarity: Similarity
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
            let mut p = Permutation::random(mat.ncols() - 1);
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
        // important, since it's possible the distorted version could have a
        // better bijection!
        let (distorted, _) = Self::permute(distorted);
        let cloud = unit_sphere_normalize(distorted);

        let similarity = similarity::polyhedron_reference_base::<false>(cloud.clone(), shape.name).expect("Reference doesn't fail");

        // let shape_rotations = shape_from_name(shape.name).generate_rotations();
        // let reference_bijection = Self::pop_centroid(bijection);
        // let found_bijection = Self::pop_centroid(similarity.bijection.clone());
        // assert!(Shape::is_rotation(&reference_bijection, &found_bijection, &shape_rotations));

        Case {shape_name: shape.name, cloud, similarity}
    }

    pub fn pass(&self, f: &dyn Fn(Matrix3N, Name) -> Result<Similarity, SimilarityError>) -> bool {
        let similarity = f(self.cloud.clone(), self.shape_name).expect("similarity fn doesn't panic");

        let shape_rotations = shape_from_name(self.shape_name).generate_rotations();
        let reference_bijection = Self::pop_centroid(self.similarity.bijection.clone());
        let found_bijection = Self::pop_centroid(similarity.bijection);
        let is_rotation = Shape::is_rotation(&reference_bijection, &found_bijection, &shape_rotations);

        let csm_close = (self.similarity.csm - similarity.csm).abs() < 1e-6;

        is_rotation && csm_close
    }
}

#[derive(Clone)]
struct Statistics<'a> {
    f: &'a dyn Fn(Matrix3N, Name) -> Result<Similarity, SimilarityError>,
    distortion_step_pass_count: Vec<usize>
}

impl<'a> Statistics<'a> {
    pub fn new(f: &'a dyn Fn(Matrix3N, Name) -> Result<Similarity, SimilarityError>) -> Statistics<'a> {
        Statistics {f, distortion_step_pass_count: Vec::new()}
    }

    pub fn step_add_passes(&mut self, passes: Vec<bool>) {
        let count = passes.into_iter().filter(|s| *s).count();
        self.distortion_step_pass_count.push(count);
    }
}

fn main() {
    let shape = &OCTAHEDRON;
    const REPETITIONS: usize = 10;

    let mut stats = [
        Statistics::new(&similarity::polyhedron_reference),
        Statistics::new(&similarity::polyhedron_base::<5, true, true>),
        Statistics::new(&similarity::polyhedron_base::<4, true, true>),
        Statistics::new(&similarity::polyhedron_base::<3, true, true>),
    ].to_vec();

    for distortion_norm in (0..=10).map(|i| 0.1 * i as f64) {
        for stat in stats.iter_mut() {
            let passes = (0..REPETITIONS).map(|_| Case::new(shape, distortion_norm).pass(stat.f)).collect();
            stat.step_add_passes(passes);
        }
    }

    for stat in stats {
        println!("{:?}", stat.distortion_step_pass_count);
    }
}
