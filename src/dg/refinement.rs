use rayon::prelude::*;
use crate::dg::{DistanceBounds, DIMENSIONS};
use thiserror::Error;
extern crate nalgebra as na;

// NOTES
// - Ideas for speedup
//   - Split up chiral sites into single-element sites and multi-element sites by an enum.
//     Calculating and storing average vectors for nearly always single atom sites everywhere is
//     probably bad
// - Ideas for parallelization
//   - Chiral constraints and fourth dimension contributions operate on
//     different sections of each gradient vector (chiral on 3d, fourth dim
//     only on 4th), so could run in parallel in shader
//   - Distance gradient incorporation may be parallelizeable further by
//     operating on distinct i-j pairs (0-1, 2-3, 4-5)

type Vector3 = na::Vector3<f64>;
type Vector4 = na::Vector4<f64>;
type DVector = na::DVector<f64>;
type VectorView3<'a> = na::Matrix<f64, na::Const<3>, na::Const<1>, na::ViewStorage<'a, f64, na::Const<3>, na::Const<1>, na::Const<1>, na::Dyn>>;
type VectorView4<'a> = na::Matrix<f64, na::Const<4>, na::Const<1>, na::ViewStorage<'a, f64, na::Const<4>, na::Const<1>, na::Const<1>, na::Dyn>>;
type VectorViewMut3<'a> = na::Matrix<f64, na::Const<3>, na::Const<1>, na::ViewStorageMut<'a, f64, na::Const<3>, na::Const<1>, na::Const<1>, na::Dyn>>;
type VectorViewMut4<'a> = na::Matrix<f64, na::Const<4>, na::Const<1>, na::ViewStorageMut<'a, f64, na::Const<4>, na::Const<1>, na::Const<1>, na::Dyn>>;

use argmin::core::{Executor, TerminationStatus, TerminationReason, IterState};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;

pub struct StrictUpperTriangleIndices {
    pub n: usize,
    pub indices: Option<(usize, usize)>
}

impl StrictUpperTriangleIndices {
    pub fn new(n: usize) -> StrictUpperTriangleIndices {
        let indices = (n > 1).then_some((0, 1));
        StrictUpperTriangleIndices {n, indices}
    }

    pub fn increment(&mut self) {
        self.indices = match self.indices {
            Some((mut i, mut j)) => {
                j += 1;

                let mut incomplete = true;
                if j == self.n {
                    i += 1;
                    j = i + 1;
                    if j == self.n {
                        incomplete = false;
                    } 
                }

                incomplete.then_some((i, j))
            },
            None => None
        };
    }

    pub fn total_len(&self) -> usize {
        (self.n.pow(2) - self.n)/ 2
    }

    pub fn linear_index(&self) -> usize {
        if let Some((i, j)) = self.indices {
            // valid indices and n > 1
            debug_assert!(i < j);
            i * (self.n - 1) - i * (i.wrapping_sub(1)) / 2 + j - 1
        } else {
            self.total_len()
        }
    }
}

impl Iterator for StrictUpperTriangleIndices {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let indices = self.indices;
        self.increment();
        indices
    }
}

impl ExactSizeIterator for StrictUpperTriangleIndices {
    fn len(&self) -> usize {
        self.total_len() - self.linear_index()
    }
}

#[derive(PartialEq, Debug)]
struct DistanceBoundGradient(Vector4);

struct DistanceBound {
    pub indices: (usize, usize),
    pub square_bounds: (f64, f64)
}

impl DistanceBound {
    pub fn error(&self, positions: &na::DVector<f64>) -> f64 {
        let (lower_squared, upper_squared) = &self.square_bounds;
        debug_assert!(lower_squared <= upper_squared);
        let (i, j) = self.indices;

        let diff = four(positions, i) - four(positions, j);
        let square_distance = diff.norm_squared();

        let upper_term = square_distance / upper_squared - 1.0;
        if upper_term > 0.0 {
            return upper_term.powi(2);
        } 

        let quotient = lower_squared + square_distance;
        let lower_term = 2.0 * lower_squared / quotient - 1.0;
        if lower_term > 0.0 {
            return lower_term.powi(2);
        }

        0.0
    }

    pub fn gradient(&self, positions: &na::DVector<f64>) -> Option<DistanceBoundGradient> {
        let (lower_squared, upper_squared) = &self.square_bounds;
        debug_assert!(lower_squared <= upper_squared);
        let (i, j) = self.indices;

        let diff = four(positions, i) - four(positions, j);
        let square_distance = diff.norm_squared();

        let upper_term = square_distance / upper_squared - 1.0;
        if upper_term > 0.0 {
            let grad = (4.0 * upper_term / upper_squared) * diff;
            return Some(DistanceBoundGradient(grad));
        } 

        let quotient = lower_squared + square_distance;
        let lower_term = 2.0 * lower_squared / quotient - 1.0;
        if lower_term > 0.0 {
            let grad = (-8.0 * lower_squared * lower_term / quotient.powi(2)) * diff;
            return Some(DistanceBoundGradient(grad));
        }

        None
    }
}

impl DistanceBoundGradient {
    fn incorporate_into(self, gradient: &mut na::DVector<f64>, indices: (usize, usize)) {
        {
            let mut part = four_mut(gradient, indices.0);
            part += self.0;
        }
        {
            let mut part = four_mut(gradient, indices.1);
            part -= self.0;
        }
    }
}

pub struct ChiralGradient(na::Matrix3x4<f64>);

pub struct Chiral {
    pub sites: [Vec<usize>; 4],
    pub adjusted_volume_bounds: (f64, f64),
    pub weight: f64
}

impl Chiral {
    pub fn target_volume_is_zero(&self) -> bool {
        let (lower, upper) = self.adjusted_volume_bounds;
        lower + upper < 1e-4
    }

    pub fn volume_positive(&self, positions: &na::DVector<f64>) -> bool {
        let [alpha, beta, gamma, delta] = array_of_ref(&self.sites)
            .map(|site| site_three(positions, site));

        let alpha_minus_delta = alpha - delta;
        let beta_minus_delta = beta - delta;
        let gamma_minus_delta = gamma - delta;

        let adjusted_volume = alpha_minus_delta.dot(&beta_minus_delta.cross(&gamma_minus_delta));
        adjusted_volume >= 0.0
    }

    pub fn error(&self, positions: &na::DVector<f64>) -> f64 {
        let (lower, upper) = self.adjusted_volume_bounds;
        let [alpha, beta, gamma, delta] = array_of_ref(&self.sites)
            .map(|site| site_three(positions, site));

        let alpha_minus_delta = alpha - delta;
        let beta_minus_delta = beta - delta;
        let gamma_minus_delta = gamma - delta;

        let adjusted_volume = alpha_minus_delta.dot(&beta_minus_delta.cross(&gamma_minus_delta));

        let term;
        if adjusted_volume < lower {
            term = self.weight * (lower - adjusted_volume);
        } else if adjusted_volume > upper {
            term = self.weight * (adjusted_volume - upper);
        } else {
            return 0.0;
        }

        term * term
    }

    pub fn gradient(&self, positions: &na::DVector<f64>) -> Option<ChiralGradient> {
        let (lower, upper) = self.adjusted_volume_bounds;
        let [alpha, beta, gamma, delta] = array_of_ref(&self.sites)
            .map(|site| site_three(positions, site));

        let alpha_minus_delta = alpha - delta;
        let beta_minus_delta = beta - delta;
        let gamma_minus_delta = gamma - delta;

        let adjusted_volume = alpha_minus_delta.dot(&beta_minus_delta.cross(&gamma_minus_delta));

        let term;
        if adjusted_volume < lower {
            term = self.weight * (adjusted_volume - lower);
        } else if adjusted_volume > upper {
            term = self.weight * (adjusted_volume - upper);
        } else {
            return None;
        }

        let factor = 2.0 * term;
        debug_assert!(factor != 0.0);

        let alpha_minus_gamma = alpha - gamma;
        let beta_minus_gamma = beta - gamma;

        let v_pairs = [
            (&beta_minus_delta,  &gamma_minus_delta),
            (&gamma_minus_delta, &alpha_minus_delta),
            (&alpha_minus_delta, &beta_minus_delta),
            (&beta_minus_gamma,  &alpha_minus_gamma)
        ];

        // There is no IntoIter implementation where the iterator takes ownership
        // of the temporary matrix like so:
        //
        // let mat = na::Matrix3x4::<f64>::from_iterator(v_pairs.into_iter()
        //     .zip(constraint.sites.iter())
        //     .flat_map(|((a, b), site)| a.cross(b).scale_mut(factor / site.len() as f64).into_iter())
        // );
        //
        // See: https://github.com/dimforge/nalgebra/issues/747
        // so we have to do things differently

        let mat = {
            let mut m = na::Matrix3x4::<f64>::from_columns(&v_pairs.map(|(a, b)| a.cross(b)));
            m.column_iter_mut()
                .zip(self.sites.iter())
                .for_each(|(mut c, site)| c *= factor / site.len() as f64);
            m
        };

        Some(ChiralGradient(mat))
    }
}

impl ChiralGradient {
    fn incorporate_into(self, gradient: &mut na::DVector<f64>, sites: &[Vec<usize>; 4]) {
        for (column, site) in self.0.column_iter().zip(sites.iter()) {
            for &i in site.iter() {
                let mut part = three_mut(gradient, i);
                part += column;
            }
        }
    }
}

pub struct SerialRefinement {
    distances: Vec<DistanceBound>,
    chirals: Vec<Chiral>,
    compress_fourth_dimension: bool
}

fn four(line: &DVector, index: usize) -> VectorView4 {
    line.fixed_view::<4, 1>(DIMENSIONS * index, 0)
}

fn four_mut(line: &mut DVector, index: usize) -> VectorViewMut4 {
    line.fixed_view_mut::<4, 1>(DIMENSIONS * index, 0)
}

fn three(line: &DVector, index: usize) -> VectorView3 {
    line.fixed_view::<3, 1>(DIMENSIONS * index, 0)
}

fn three_mut(line: &mut DVector, index: usize) -> VectorViewMut3 {
    line.fixed_view_mut::<3, 1>(DIMENSIONS * index, 0)
}

// fn site_four(line: &DVector, site: &Vec<usize>) -> Vector4 {
//     if site.len() == 1 {
//         return four(line, site[0]).into();
//     }
// 
//     let sum = site.iter()
//         .fold(na::Vector4::zeros(), |acc, &index| acc + four(line, index));
//     sum / site.len() as f64
// }

// TODO try alteration where return type is an enum of Vector3 and VectorView3
// or where site is a smallvec
fn site_three(line: &DVector, site: &Vec<usize>) -> Vector3 {
    if site.len() == 1 {
        return three(line, site[0]).into();
    }

    let sum = site.iter()
        .fold(na::Vector3::zeros(), |acc, &index| acc + three(line, index));
    sum / site.len() as f64
}

fn negate_y_coordinates(line: &mut DVector) {
    let n = line.len() / DIMENSIONS;
    line.view_with_steps_mut((1, 0), (n, 1), (DIMENSIONS - 1, 0)).neg_mut();
}

/// Polyfill for array::each_ref while awaiting stabilization, see
/// https://github.com/rust-lang/rust/issues/76118
fn array_of_ref<T, const N:usize>(arr: &[T;N])->[&T;N] {
    use core::mem::MaybeUninit;
    let mut out:MaybeUninit<[&T;N]> = MaybeUninit::uninit();
    
    let buf = out.as_mut_ptr() as *mut &T;
    let mut refs = arr.iter();
    
    for i in 0..N {
        unsafe { buf.add(i).write(refs.next().unwrap()) }
    }
    
    unsafe { out.assume_init() }
}

fn linearize_bounds(bounds: DistanceBounds) -> Vec<DistanceBound> {
    let n = bounds.n();
    let bounds_squared = bounds.take_matrix().map(|v| v.powi(2));
    StrictUpperTriangleIndices::new(n)
        .map(|(i, j)| DistanceBound {
            indices: (i, j),
            square_bounds: (bounds_squared[(j, i)], bounds_squared[(i, j)]),
        })
        .collect()
}

impl SerialRefinement {
    pub fn new(bounds: DistanceBounds, chirals: Vec<Chiral>) -> SerialRefinement {
        SerialRefinement {distances: linearize_bounds(bounds), chirals, compress_fourth_dimension: false}
    }

    pub fn distance_error(&self, positions: &na::DVector<f64>) -> f64 {
        self.distances.iter()
            .map(|bound| bound.error(positions))
            .sum()
    }

    pub fn distance_gradient(&self, positions: &na::DVector<f64>) -> DVector {
        let mut gradient: na::DVector<f64> = na::DVector::zeros(positions.nrows());

        for bound in self.distances.iter() {
            if let Some(contribution) = bound.gradient(positions) {
                contribution.incorporate_into(&mut gradient, bound.indices);
            }
        }

        gradient
    }

    pub fn chiral_error(&self, positions: &na::DVector<f64>) -> f64 {
        self.chirals.iter()
            .map(|constraint| constraint.error(positions))
            .sum()
    }

    pub fn chiral_gradient(&self, positions: &na::DVector<f64>) -> DVector {
        let mut gradient: na::DVector<f64> = na::DVector::zeros(positions.nrows());

        for constraint in self.chirals.iter() {
            if let Some(contribution) = constraint.gradient(positions) {
                contribution.incorporate_into(&mut gradient, &constraint.sites);
            }
        }

        gradient
    }

    pub fn fourth_dimension_error(&self, positions: &na::DVector<f64>) -> f64 {
        if !self.compress_fourth_dimension {
            return 0.0;
        }

        let n = positions.len() / DIMENSIONS;
        (0..n).map(|i| positions[DIMENSIONS * i + 3].powi(2))
            .sum()
    }

    pub fn fourth_dimension_gradient(&self, positions: &na::DVector<f64>, mut gradient: DVector) -> DVector {
        if !self.compress_fourth_dimension {
            return gradient;
        }

        let n = positions.len() / DIMENSIONS;
        for i in 0..n {
            gradient[DIMENSIONS * i + 3] += 2.0 * positions[DIMENSIONS * i + 3];
        }
        gradient
    }
}

impl RefinementErrorFunction for SerialRefinement {
    fn error(&self, positions: &DVector) -> f64 {
        self.distance_error(positions)
            + self.chiral_error(positions)
            + self.fourth_dimension_error(positions)
    }

    fn gradient(&self, positions: &DVector) -> DVector {
        let gradient = self.distance_gradient(positions) + self.chiral_gradient(positions);
        self.fourth_dimension_gradient(positions, gradient)
    }

    fn set_4d_compression(&mut self) {
        self.compress_fourth_dimension = true;
    }
}

pub struct ParallelRefinement {
    distances: Vec<DistanceBound>,
    chirals: Vec<Chiral>,
    compress_fourth_dimension: bool
}

impl ParallelRefinement {
    pub fn new(bounds: DistanceBounds, chirals: Vec<Chiral>) -> Self {
        Self {distances: linearize_bounds(bounds), chirals, compress_fourth_dimension: false}
    }

    pub fn distance_error(&self, positions: &na::DVector<f64>) -> f64 {
        self.distances.par_iter()
            .map(|bound| bound.error(positions))
            .sum()
    }

    pub fn distance_gradient(&self, positions: &na::DVector<f64>) -> DVector {
        // Calculate gradient contributions for each bound
        let contributions: Vec<Option<DistanceBoundGradient>> = self.distances.par_iter()
            .map(|bound| bound.gradient(positions))
            .collect();

        // Serially collect the gradient
        let mut gradient: na::DVector<f64> = na::DVector::zeros(positions.nrows());
        for (maybe_contribution, bound) in contributions.into_iter().zip(self.distances.iter()) {
            if let Some(contribution) = maybe_contribution {
                contribution.incorporate_into(&mut gradient, bound.indices);
            }
        }
        gradient
    }

    pub fn chiral_error(&self, positions: &na::DVector<f64>) -> f64 {
        self.chirals.par_iter()
            .map(|constraint| constraint.error(positions))
            .sum()
    }

    pub fn chiral_gradient(&self, positions: &na::DVector<f64>) -> DVector {
        // Calculate chiral gradient contributions in parallel
        let contributions: Vec<Option<ChiralGradient>> = self.chirals.par_iter()
            .map(|constraint| constraint.gradient(positions))
            .collect();

        // Serially collect the gradient contributions
        let mut gradient: na::DVector<f64> = na::DVector::zeros(positions.nrows());
        for (constraint, maybe_contribution) in self.chirals.iter().zip(contributions.into_iter()) {
            if let Some(contribution) = maybe_contribution {
                contribution.incorporate_into(&mut gradient, &constraint.sites);
            }
        }
        gradient
    }

    pub fn fourth_dimension_error(&self, positions: &na::DVector<f64>) -> f64 {
        if !self.compress_fourth_dimension {
            return 0.0;
        }

        let n = positions.len() / DIMENSIONS;
        (0..n).into_par_iter()
            .map(|i| positions[DIMENSIONS * i + 3].powi(2))
            .sum()
    }

    pub fn fourth_dimension_gradient(&self, positions: &na::DVector<f64>, gradient: DVector) -> DVector {
        if !self.compress_fourth_dimension {
            return gradient;
        }

        // Positions could be reshaped for a nicer form
        //
        //  let matrix_grad = gradient.reshape_generic(na::Const<DIMENSIONS>, na::Dyn(n));
        //  matrix_grad.par_column_iter_mut()
        //      .zip(positions.par_column_iter())
        //      .for_each(|(mut grad_v, pos_v)| grad_v.w() += 2.0 * pos_v.w(););

        let n = positions.len() / DIMENSIONS;
        let mut matrix_grad = gradient.reshape_generic(na::Const::<DIMENSIONS>, na::Dyn(n));
        matrix_grad.par_column_iter_mut()
            .zip((0..n).into_par_iter())
            .for_each(|(mut grad_v, i)| {grad_v[3] += 2.0 * positions[DIMENSIONS * i + 3];});

        matrix_grad.reshape_generic(na::Dyn(n * DIMENSIONS), na::Const::<1>)
    }
}

impl RefinementErrorFunction for ParallelRefinement {
    fn error(&self, positions: &DVector) -> f64 {
        self.distance_error(positions)
            + self.chiral_error(positions)
            + self.fourth_dimension_error(positions)
    }

    fn gradient(&self, positions: &DVector) -> DVector {
        let gradient = self.distance_gradient(positions) + self.chiral_gradient(positions);
        self.fourth_dimension_gradient(positions, gradient)
    }

    fn set_4d_compression(&mut self) {
        self.compress_fourth_dimension = true;
    }
}

pub trait RefinementErrorFunction {
    fn error(&self, positions: &DVector) -> f64;
    fn gradient(&self, positions: &DVector) -> DVector;
    fn set_4d_compression(&mut self);
}

impl argmin::core::CostFunction for &dyn RefinementErrorFunction {
    type Param = DVector;
    type Output = f64;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.error(param))
    }
}

impl argmin::core::Gradient for &dyn RefinementErrorFunction {
    type Param = DVector;
    type Gradient = DVector;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok((*self).gradient(param))
    }
}

pub struct Refinement {
    pub coords: na::Matrix3xX<f64>,
    pub steps: usize
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum RefinementError {
    #[error("Solver failure")]
    SolverFailure,
}

fn check_termination_status<P, G, J, H, F>(state: &IterState<P, G, J, H, F>) -> Result<(), RefinementError> {
    match &state.termination_status {
        TerminationStatus::Terminated(reason) => match reason {
            TerminationReason::TargetCostReached | TerminationReason::SolverConverged => Ok(()),
            _ => Err(RefinementError::SolverFailure)
        },
        TerminationStatus::NotTerminated => {panic!("Unterminated terminated solver");}
    }
}

pub fn refine(mut problem: impl RefinementErrorFunction, positions: na::Matrix4xX<f64>) -> Result<Refinement, RefinementError> {
    const LBFGS_MEMORY: usize = 32;

    let n = positions.len();
    let linear_positions = positions.reshape_generic(na::Dyn(n), na::Const::<1>);
    // TODO invert y if beneficial to chirals

    let linesearch = MoreThuenteLineSearch::new();

    // Minimize distance and chiral constraints
    let solver = LBFGS::new(linesearch.clone(), LBFGS_MEMORY).with_tolerance_grad(1e-6).unwrap();
    let mut result = Executor::new(&problem as &dyn RefinementErrorFunction, solver)
        .configure(|state| state.param(linear_positions))
        .run()
        .unwrap();
    check_termination_status(&result.state)?;
    let uncompressed_positions = result.state.take_best_param()
        .expect("Successful termination implies optimal parameters present");
    let relaxation_steps = result.state.iter;

    // Compress out fourth dimension
    problem.set_4d_compression();
    let solver = LBFGS::new(linesearch, LBFGS_MEMORY).with_tolerance_grad(1e-6).unwrap();
    let mut result = Executor::new(&problem as &dyn RefinementErrorFunction, solver)
        .configure(|state| state.param(uncompressed_positions))
        .run()
        .unwrap();
    check_termination_status(&result.state)?;
    let final_positions = result.state.take_best_param()
        .expect("Successful termination implies optimal parameters present");
    let compression_steps = result.state.iter;
    let steps: usize = (relaxation_steps + compression_steps).try_into().unwrap();

    // Reshape matrix and drop fourth dimension
    let matrix_positions = final_positions.reshape_generic(na::Const::<DIMENSIONS>, na::Dyn(n / DIMENSIONS));
    let coords = matrix_positions.remove_row(DIMENSIONS - 1);

    Ok(Refinement {coords, steps})
}

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::dg::{DistanceMatrix, MetricMatrix, MetrizationPartiality};
    use crate::dg::refinement::*;
    use crate::shapes::TETRAHEDRON;

    extern crate nalgebra as na;
    use num_traits::float::Float;

    #[test]
    fn index_iterator() {
        for i in 0..6 {
            assert_eq!(StrictUpperTriangleIndices::new(i).linear_index(), 0);

            assert_eq!(
                StrictUpperTriangleIndices::new(i).len(),
                StrictUpperTriangleIndices::new(i).count()
            );

            itertools::assert_equal(
                StrictUpperTriangleIndices::new(i), 
                (0..i).combinations(2).map(|v| (v[0], v[1]))
            );
        }
    }

    /// Central difference numerical gradient
    fn numerical_gradient<F: Float + na::Scalar>(cost: &dyn Fn(&na::DVector<F>) -> F, args: &na::DVector<F>) -> na::DVector<F> {
        let n = args.len();
        let mut params = args.clone();
        let h = F::from(1e-4).unwrap();
        na::DVector::<F>::from_iterator(
            n,
            (0..n).map(|i| {
                let arg = args[i];
                params[i] = arg + h;
                let b = cost(&params);
                params[i] = arg - h;
                let a = cost(&params);
                params[i] = arg;
                (b - a) / (F::from(2.0).unwrap() * h)
            })
        )
    }

    #[test]
    fn distance_bound_gradient() {
        let positions: na::DVector<f64> = na::DVector::new_random(8);
        let r = (four(&positions, 0) - four(&positions, 1)).norm_squared();
        let indices = (0, 1);

        let h = 0.01 * r;

        // Within bounds
        let bound = DistanceBound {
            indices,
            square_bounds: (r - h, r + h)
        };
        assert_eq!(bound.error(&positions), 0.0);
        assert_eq!(bound.gradient(&positions), None);

        // Below bounds
        let short = 0.9 * r;
        let bound = DistanceBound {
            indices,
            square_bounds: (short - h, short + h)
        };
        assert!(bound.error(&positions) > 0.0);
        let maybe_analytical_gradient = bound.gradient(&positions);
        assert!(maybe_analytical_gradient.is_some());
        let mut analytical_gradient = na::DVector::<f64>::zeros(8);
        maybe_analytical_gradient.unwrap().incorporate_into(&mut analytical_gradient, bound.indices);
        approx::assert_relative_eq!(
            analytical_gradient,
            numerical_gradient(&|param| bound.error(param), &positions),
            epsilon=1e-7
        );

        // Above bounds
        let long = 1.1 * r;
        let bound = DistanceBound {
            indices,
            square_bounds: (long - h, long + h)
        };
        assert!(bound.error(&positions) > 0.0);
        let maybe_analytical_gradient = bound.gradient(&positions);
        assert!(maybe_analytical_gradient.is_some());
        let mut analytical_gradient = na::DVector::<f64>::zeros(8);
        maybe_analytical_gradient.unwrap().incorporate_into(&mut analytical_gradient, bound.indices);
        approx::assert_relative_eq!(
            analytical_gradient,
            numerical_gradient(&|param| bound.error(param), &positions),
            epsilon=1e-7
        );
    }

    #[test]
    fn chiral_bound_gradient() {
        let shape = &TETRAHEDRON;
        let bounds = crate::dg::modeling::solitary_shape::shape_into_bounds(shape);
        let distances = DistanceMatrix::try_from_distance_bounds(bounds, MetrizationPartiality::Complete).expect("Successful metrization");
        let metric = MetricMatrix::from_distance_matrix(distances);
        let coords = metric.embed();
        let n = coords.len();
        let mut linear_coords = coords.reshape_generic(na::Dyn(n), na::Const::<1>);
        let chirals: Vec<Chiral> = shape.find_tetrahedra().into_iter()
            .map(|tetr| crate::dg::modeling::solitary_shape::chiral_from_tetrahedron(tetr, shape, 0.1))
            .collect();

        if let Some(chiral) = chirals.first() {
            // Ensure sign of volume for chiral constraint is correct
            if !chiral.volume_positive(&linear_coords) {
                negate_y_coordinates(&mut linear_coords);
            }

            assert!(chiral.volume_positive(&linear_coords));

            if chiral.error(&linear_coords) > 0.0 {
                let mut analytical_gradient = na::DVector::<f64>::zeros(n);
                if let Some(contribution) = chiral.gradient(&linear_coords) {
                    contribution.incorporate_into(&mut analytical_gradient, &chiral.sites);
                }

                approx::assert_relative_eq!(
                    analytical_gradient,
                    numerical_gradient(&|p| chiral.error(p), &linear_coords),
                    epsilon=1e-7
                );
            }

            // Ensure sign of volume for constraint is wrong
            negate_y_coordinates(&mut linear_coords);
            assert!(chiral.error(&linear_coords) > 0.0);
            let mut analytical_gradient = na::DVector::<f64>::zeros(n);
            if let Some(contribution) = chiral.gradient(&linear_coords) {
                contribution.incorporate_into(&mut analytical_gradient, &chiral.sites);
            }

            approx::assert_relative_eq!(
                analytical_gradient,
                numerical_gradient(&|p| chiral.error(p), &linear_coords),
                epsilon=1e-7
            );
        }
    }
}
