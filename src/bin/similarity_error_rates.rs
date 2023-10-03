use molassembler::strong::{Index, IndexBase};
use molassembler::strong::matrix::Positions;
use molassembler::shapes::similarity::{skip_vertices, polyhedron_reference, unit_sphere_normalize, SimilarityError};
use molassembler::quaternions::Matrix3N;
use molassembler::permutation::{Permutation, Permutatable};
use molassembler::strong::bijection::{Bijection, Bijectable};
use molassembler::quaternions::random_rotation;
use molassembler::shapes::*;

use statrs::statistics::Statistics as OtherStatistics;
use std::collections::{HashMap, HashSet};
use itertools::Itertools;

// NOTES
// - Trying all close-to-optimal sub-permutations generated in linear assignment 
//   helps mitigate some failure cases.
//   - Pessimizes average case by tracking multiple partial fits and performing multiple quaternion
//     fits for each prematch permutation. Also requires brute force linear assignment algorithm,
//     since near-optimal permutations are not enumerated or visited by jv.
//   - At large distortions and few prematches, the linear assignment approximation breaks down
//     hard enough that an arbitrary well depth threshold of 1.0 cannot mitigate all failures
// - Better to forward to a higher prematch count if the difference between partial fit and full
//   fit is particularly bad (dependent on current prematch count)
//   

pub struct SimilarityAnalysis {
    /// Best bijection between shape vertices and input matrix
    pub bijection: Bijection<Vertex, Column>,
    /// Continuous shape measure
    pub csm: f64,
    /// Deviation between partial and full quaternion fit msds
    pub msd_dev: f64
}

pub fn polyhedron_analysis<const PREMATCH: usize, const USE_SKIPS: bool, const LAP_JV: bool>(x: Matrix3N, shape: &Shape) -> Result<SimilarityAnalysis, SimilarityError> {
    const MIN_PREMATCH: usize = 2;
    assert!(MIN_PREMATCH <= PREMATCH); // TODO trigger compilation failure? static assert?

    type Occupation = Bijection<Vertex, Column>;
    let n = x.ncols();
    if n != shape.num_vertices() + 1 {
        return Err(SimilarityError::ParticleNumberMismatch);
    }

    if n <= PREMATCH {
        return polyhedron_reference(x, shape)
            .map(|s| SimilarityAnalysis {bijection: s.bijection, csm: s.csm, msd_dev: 0.0});
    }

    let cloud = Positions::<Column>::new(unit_sphere_normalize(x));

    // Add centroid to shape coordinates and normalize
    let shape_coordinates = shape.coordinates.matrix.clone().insert_column(n - 1, 0.0);
    let shape_coordinates = unit_sphere_normalize(shape_coordinates);
    let shape_coordinates = Positions::<Vertex>::new(shape_coordinates);

    type PartialPermutation = HashMap<Column, Vertex>;

    struct PartialMsd {
        msd: f64,
        mapping: PartialPermutation,
        partial_msd: f64,
    }
    let left_free: Vec<Column> = (PREMATCH..n).map(Column::from).collect();

    let skips = USE_SKIPS.then(|| skip_vertices(shape));
    let narrow = Vertex::range(n)
        .permutations(PREMATCH)
        .filter(|vertices| {
            match USE_SKIPS {
                true => !skips.as_ref().unwrap()[(vertices[0].get(), vertices[1].get())],
                false => true
            }
        })
        .fold(
            PartialMsd {msd: f64::MAX, mapping: PartialPermutation::new(), partial_msd: f64::MAX},
            |best, vertices| -> PartialMsd {
                let mut partial_map = PartialPermutation::with_capacity(n);
                vertices.iter().enumerate().for_each(|(c, v)| { partial_map.insert(Column::from(c), *v); });
                let partial_fit = cloud.quaternion_fit_map(&shape_coordinates, &partial_map);

                // If the msd caused only by the partial map is already worse, skip
                if partial_fit.msd > best.msd {
                    return best;
                }

                // Assemble cost matrix of matching each pair of unmatched vertices
                let right_free: Vec<Vertex> = Vertex::range(n)
                    .filter(|v| !vertices.contains(v))
                    .collect();

                let v = left_free.len();
                let prematch_rotated_shape = partial_fit.rotate_rotor(&shape_coordinates.matrix.clone());
                let prematch_rotated_shape = Positions::<Vertex>::new(prematch_rotated_shape);

                let cost_fn = |i, j| (cloud.point(left_free[i]) - prematch_rotated_shape.point(right_free[j])).norm_squared();
                let sub_permutation = match LAP_JV {
                    true => similarity::linear_assignment::optimal(v, &cost_fn),
                    false => similarity::linear_assignment::brute_force(v, &cost_fn)
                };

                // Fuse pre-match and best subpermutation
                for (i, j) in sub_permutation.iter() {
                    partial_map.insert(left_free[i], right_free[*j]);
                }

                // Make a clean quaternion fit with the full mapping
                let full_fit = cloud.quaternion_fit_map(&shape_coordinates, &partial_map);
                if full_fit.msd < best.msd {
                    PartialMsd {msd: full_fit.msd, mapping: partial_map, partial_msd: partial_fit.msd}
                } else {
                    best
                }
            }
        );

    let msd_dev = narrow.msd - narrow.partial_msd;
    let is_bad_approximation = {
        let refit_delta_threshold = match PREMATCH {
            2 => 0.25,
            3 => 0.375,
            4 => 0.425,
            _ => 0.5
        };
        msd_dev > refit_delta_threshold
    };
    if is_bad_approximation {
        return match PREMATCH {
            2 => polyhedron_analysis::<3, USE_SKIPS, LAP_JV>(cloud.matrix, shape),
            3 => polyhedron_analysis::<4, USE_SKIPS, LAP_JV>(cloud.matrix, shape),
            4 => polyhedron_analysis::<5, USE_SKIPS, LAP_JV>(cloud.matrix, shape),
            _ => similarity::polyhedron_reference(cloud.matrix, shape)
                    .map(|s| SimilarityAnalysis {bijection: s.bijection, csm: s.csm, msd_dev: 0.0})
        };
    }

    /* Given the best permutation for the rotational fit, we still have to
     * minimize over the isotropic scaling factor. It is cheaper to reorder the
     * positions once here for the minimization so that memory access is in-order
     * during the repeated scaling minimization function calls.
     *
     * NOTE: The bijection is inverted here
     */
    let best_bijection = {
        let mut sigma: Vec<_> = (0..n).collect();
        narrow.mapping.iter().for_each(|(c, v)| { sigma[v.get()] = c.get(); });
        Occupation::new(Permutation::try_from(sigma).expect("Valid permutation"))
    };

    if cfg!(debug_assertions) {
        for (c, v) in narrow.mapping.iter() {
            assert_eq!(best_bijection.get(v), Some(*c));
        }
    }

    let permuted_shape = shape_coordinates.biject(&best_bijection).expect("Matching size");
    let fit = cloud.quaternion_fit_rotor(&permuted_shape);
    let rotated_shape = fit.rotate_rotor(&permuted_shape.matrix);

    let csm = similarity::scaling::optimize_csm(&cloud.matrix, &rotated_shape);
    Ok(SimilarityAnalysis {bijection: best_bijection, csm, msd_dev})
}

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
            p.push();
            Bijection::new(p)
        };

        let permuted = mat.permute(&bijection.permutation).expect("Matching size");

        (permuted, bijection)
    }

    fn pop_centroid(mut bijection: Bijection<Vertex, Column>) -> Bijection<Vertex, Column> {
        let _ = bijection.permutation.pop_if_fixed_point().expect("No zero-length bijections");
        bijection
    }

    pub fn new(shape: &Shape, distortion_norm: f64) -> Case {
        let coords = shape.coordinates.matrix.clone().insert_column(shape.num_vertices(), 0.0);
        let distorted = Self::distort(Self::rotate(coords), distortion_norm);
        // The bijection with which the distorted version is generated is not
        // necessarily useful for comparison since it's possible the distorted
        // version could have a better bijection!
        let (distorted, bijection) = Self::permute(distorted);
        let cloud = unit_sphere_normalize(distorted);

        Case {shape_name: shape.name, cloud, expected_bijection: bijection}
    }

    pub fn pass(&self, f: &dyn Fn(Matrix3N, &Shape) -> Result<SimilarityAnalysis, SimilarityError>, rotations: &HashSet<molassembler::shapes::Rotation>) -> (bool, f64) {
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

            (csm_close && is_rotation, f_similarity.msd_dev)
        } else {
            (true, f_similarity.msd_dev)
        }
    }
}

#[derive(Clone)]
struct Statistics<'a> {
    f: &'a dyn Fn(Matrix3N, &Shape) -> Result<SimilarityAnalysis, SimilarityError>,
    name: String,
    distortion_failures: HashMap<usize, usize>
}

impl<'a> Statistics<'a> {
    pub fn new(f: &'a dyn Fn(Matrix3N, &Shape) -> Result<SimilarityAnalysis, SimilarityError>, name: impl Into<String>) -> Statistics<'a> {
        Statistics {f, name: name.into(), distortion_failures: HashMap::new()}
    }
}

fn flatten(map: &HashMap<usize, usize>) -> Vec<usize> {
    Vec::from_iter(map.keys().sorted().map(|k| map[k]))
}

const DISTORTION_START: usize = 6;

fn main() {
    println!("Distortion range: [0.{}, 1.0]", DISTORTION_START);

    for (shape_size, shapes) in &SHAPES.iter().group_by(|s| s.num_vertices()) {
        if !(6..10).contains(&shape_size) {
            continue;
        }

        const REPETITIONS: usize = 200;

        let mut stats = [
            Statistics::new(&polyhedron_analysis::<5, true, true>, "prematch(5)"),
            Statistics::new(&polyhedron_analysis::<4, true, true>, "prematch(4)"),
            Statistics::new(&polyhedron_analysis::<3, true, true>, "prematch(3)"),
            Statistics::new(&polyhedron_analysis::<2, true, true>, "prematch(2)"),
        ].to_vec();

        let mut shape_count = 0;
        for shape in shapes {
            let rotations = shape.generate_rotations();

            for i in DISTORTION_START..=10 {
                let distortion_norm = 0.1 * i as f64;

                for stat in stats.iter_mut() {
                    struct Accumulator {
                        failures: usize,
                        msd_devs: (Vec<f64>, Vec<f64>),
                    }

                    let accumulated = (0..REPETITIONS)
                        .map(|_| Case::new(shape, distortion_norm).pass(stat.f, &rotations))
                        .fold(
                            Accumulator {failures: 0, msd_devs: (Vec::new(), Vec::new())},
                            |mut acc, (pass, msd_dev)| {
                                if pass {
                                    acc.msd_devs.0.push(msd_dev);
                                } else {
                                    acc.failures += 1;
                                    acc.msd_devs.1.push(msd_dev);
                                }

                                acc
                            }
                        );

                    if let Some(existing_failures) = stat.distortion_failures.get_mut(&i) {
                        *existing_failures += accumulated.failures;
                    } else {
                        stat.distortion_failures.insert(i, accumulated.failures);
                    }

                    if accumulated.failures > 0 {
                        println!("{:.2} w/ {}: success dev {:.2}. Failure dev {:.2} ({})", 
                                 distortion_norm,
                                 stat.name,
                                 accumulated.msd_devs.0.mean(),
                                 accumulated.msd_devs.1.mean(),
                                 accumulated.failures
                                 );
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
