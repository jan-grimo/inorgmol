use rayon::prelude::*;
use itertools::Itertools;
use crate::dg::{DistanceBounds, DIMENSIONS};
use thiserror::Error;
extern crate nalgebra as na;

/// Float for use in Distance Geometry
pub trait Float: argmin::core::ArgminFloat
    + num_traits::NumAssign 
    + num_traits::NumAssignRef 
    + num_traits::NumRef 
    + na::RealField 
    + std::iter::Sum 
    + argmin_math::ArgminZero 
{}

impl<F> Float for F where F: argmin::core::ArgminFloat
    + num_traits::NumAssign 
    + num_traits::NumAssignRef 
    + num_traits::NumRef 
    + na::RealField 
    + std::iter::Sum 
    + argmin_math::ArgminZero 
{}

// Ensure f32 and f64 implement Float
const _: () = {
    fn assert_float<F: Float>() {}
    fn assert_all() {
        assert_float::<f32>();
        assert_float::<f64>();
    }
};

// NOTES
// - Ideas for speedup
//   - Split up chiral sites into single-element sites and multi-element sites by an enum.
//     Calculating and storing average vectors for nearly always single atom sites everywhere is
//     probably bad
// - Ideas for parallelization
//   - Chiral constraints and fourth dimension contributions operate on
//     different sections of each gradient vector (chiral on 3d, fourth dim
//     only on 4th), so could run in parallel in a shader

type View<'a, T, R, C> = na::Matrix<T, R, C, na::ViewStorage<'a, T, R, C, na::Const<1>, na::Dyn>>;
type VectorView3<'a, T> = View<'a, T, na::Const<3>, na::Const<1>>;
type VectorView4<'a, T> = View<'a, T, na::Const<4>, na::Const<1>>;

type ViewMut<'a, T, R, C> = na::Matrix<T, R, C, na::ViewStorageMut<'a, T, R, C, na::Const<1>, na::Dyn>>;
type VectorViewMut3<'a, T> = ViewMut<'a, T, na::Const<3>, na::Const<1>>;
type VectorViewMut4<'a, T> = ViewMut<'a, T, na::Const<4>, na::Const<1>>;

use argmin::core::{Executor, TerminationStatus, TerminationReason, IterState};
use argmin::solver::quasinewton::LBFGS;
use argmin::solver::linesearch::MoreThuenteLineSearch;

/// Gradient contribution of a distance bound
#[derive(PartialEq, Debug)]
pub struct DistanceBoundGradient<F: Float>(na::Vector4<F>);

/// Two-particle distance bound
#[derive(Clone)]
pub struct DistanceBound<F: Float> {
    /// Particle indices (i, j) with i < j
    pub indices: (usize, usize),
    /// Squared distance bounds (lower, upper)
    pub square_bounds: (F, F)
}

impl<F: Float> DistanceBound<F> {
    /// Calculate error contribution
    pub fn error(&self, positions: &na::DVector<F>) -> F {
        let (lower_squared, upper_squared) = &self.square_bounds;
        debug_assert!(lower_squared <= upper_squared);
        let (i, j) = self.indices;

        let diff = four(positions, i) - four(positions, j);
        let square_distance = diff.norm_squared();

        let upper_term = square_distance / upper_squared - F::one();
        if upper_term > <F as num_traits::Zero>::zero() {
            return num_traits::Float::powi(upper_term, 2);
        } 

        let quotient = square_distance + lower_squared;
        let lower_term = F::from(2.0).unwrap() * lower_squared / quotient - F::one();
        if lower_term > <F as num_traits::Zero>::zero() {
            return num_traits::Float::powi(lower_term, 2);
        }

        <F as num_traits::Zero>::zero()
    }

    /// Calculate gradient contribution, if any
    pub fn gradient(&self, positions: &na::DVector<F>) -> Option<DistanceBoundGradient<F>> {
        let (lower_squared, upper_squared) = &self.square_bounds;
        debug_assert!(lower_squared <= upper_squared);
        let (i, j) = self.indices;

        let diff = four(positions, i) - four(positions, j);
        let square_distance = diff.norm_squared();

        let upper_term = square_distance / upper_squared - F::one();
        if upper_term > <F as num_traits::Zero>::zero() {
            let grad = diff.scale(F::from(4.0).unwrap() * upper_term / upper_squared);
            return Some(DistanceBoundGradient(grad));
        } 

        let quotient = square_distance + lower_squared;
        let lower_term = F::from(2.0).unwrap() * lower_squared / quotient - F::one();
        if lower_term > <F as num_traits::Zero>::zero() {
            let grad = diff.scale(F::from(-8.0).unwrap() * lower_squared * lower_term / num_traits::Float::powi(quotient, 2));
            return Some(DistanceBoundGradient(grad));
        }

        None
    }
}

impl<F: Float> DistanceBoundGradient<F> {
    /// Incorporate the distance bound gradient contribution into the gradient
    fn incorporate_into(self, gradient: &mut na::DVector<F>, indices: (usize, usize)) {
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

/// Chiral constraint gradient contribution data
pub struct ChiralGradient<F: Float>(na::Matrix3x4<F>);

/// Chiral constraint
#[derive(Clone)]
pub struct Chiral<F: Float> {
    /// Particle indices for each vertex of tetrahedron
    pub sites: [Vec<usize>; 4],
    /// Adjusted volume bounds (lower, upper) with V' = 6 V
    pub adjusted_volume_bounds: (F, F),
    /// Weight of constraint (exists to weaken flattening constraints)
    pub weight: F
}

impl<F: Float> Chiral<F> {
    /// Tests if constraint volume bounds accept zero volume
    pub fn target_volume_is_zero(&self) -> bool {
        let (lower, upper) = self.adjusted_volume_bounds;
        lower + upper < F::from(1e-4).unwrap()
    }

    /// Tests if volume of tetrahedron spanned by constraint's sites positive
    pub fn volume_positive(&self, positions: &na::DVector<F>) -> bool {
        let [alpha, beta, gamma, delta] = array_of_ref(&self.sites)
            .map(|site| site_three(positions, site));

        let alpha_minus_delta = alpha - delta;
        let beta_minus_delta = beta - delta;
        let gamma_minus_delta = gamma - delta;

        let adjusted_volume = alpha_minus_delta.dot(&beta_minus_delta.cross(&gamma_minus_delta));
        adjusted_volume >= <F as num_traits::Zero>::zero()
    }

    /// Calculate error contribution
    pub fn error(&self, positions: &na::DVector<F>) -> F {
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
            return <F as num_traits::Zero>::zero();
        }

        term * term
    }

    /// Calculate gradient contribution, if any
    pub fn gradient(&self, positions: &na::DVector<F>) -> Option<ChiralGradient<F>> {
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

        let factor = F::from(2.0).unwrap() * term;
        debug_assert!(factor != <F as num_traits::Zero>::zero());

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
            let mut m = na::Matrix3x4::<F>::from_columns(&v_pairs.map(|(a, b)| a.cross(b)));
            m.column_iter_mut()
                .zip(self.sites.iter())
                .for_each(|(mut c, site)| c *= factor / F::from(site.len()).unwrap());
            m
        };

        Some(ChiralGradient(mat))
    }
}

impl<F: Float> ChiralGradient<F> {
    /// Incorporate gradient contribution of chiral constraint into gradient
    fn incorporate_into(self, gradient: &mut na::DVector<F>, sites: &[Vec<usize>; 4]) {
        for (column, site) in self.0.column_iter().zip(sites.iter()) {
            for &i in site.iter() {
                let mut part = three_mut(gradient, i);
                part += column;
            }
        }
    }
}

/// Stage of refinement
pub enum Stage {
    /// Invert chiral constraints so they have positive volume
    FixChirals,
    /// Compress out fourth spatial dimension
    CompressFourthDimension
}

/// Distance and chiral constraints for refinement
#[derive(Clone)]
pub struct Bounds<F: Float> {
    /// Distance bounds
    pub distances: Vec<DistanceBound<F>>,
    /// Chiral bounds
    pub chirals: Vec<Chiral<F>>,
}

impl<F: Float> Bounds<F> {
    /// Construct from a distance bound matrix and list of chiral constraints
    pub fn new(bounds: DistanceBounds, chirals: Vec<Chiral<F>>) -> Bounds<F> {
        Bounds {distances: Self::linearize_bounds(bounds), chirals}
    }

    /// Reform bounds matrix into individual objects in linear order
    fn linearize_bounds(bounds: DistanceBounds) -> Vec<DistanceBound<F>> {
        let n = bounds.n();
        let bounds_squared = bounds.take_matrix().map(|v| v.powi(2));
        (0..n).tuple_combinations()
            .map(|(i, j)| DistanceBound::<F> {
                indices: (i, j),
                square_bounds: (F::from(bounds_squared[(j, i)]).unwrap(), F::from(bounds_squared[(i, j)]).unwrap()),
            })
            .collect()
    }
}

/// Serial computation of error function
pub struct Serial<F: Float> {
    bounds: Bounds<F>,
    stage: Stage
}

fn four<F>(line: &na::DVector<F>, index: usize) -> VectorView4<F> {
    line.fixed_view::<4, 1>(DIMENSIONS * index, 0)
}

/// Mutable four-dimensional view of linearized positions
pub(crate) fn four_mut<F>(line: &mut na::DVector<F>, index: usize) -> VectorViewMut4<F> {
    line.fixed_view_mut::<4, 1>(DIMENSIONS * index, 0)
}

fn three<F>(line: &na::DVector<F>, index: usize) -> VectorView3<F> {
    line.fixed_view::<3, 1>(DIMENSIONS * index, 0)
}

fn three_mut<F>(line: &mut na::DVector<F>, index: usize) -> VectorViewMut3<F> {
    line.fixed_view_mut::<3, 1>(DIMENSIONS * index, 0)
}

// TODO try alteration where return type is an enum of Vector3 and VectorView3
// or where site is a smallvec
fn site_three<F: Float>(line: &na::DVector<F>, site: &Vec<usize>) -> na::Vector3<F> {
    if site.len() == 1 {
        return three(line, site[0]).into();
    }

    let sum = site.iter()
        .fold(na::Vector3::zeros(), |acc, &index| acc + three(line, index));
    sum / F::from(site.len()).expect("Initialization from integer is fine")
}

/// Invert the y coordinates of linearized coordinates
pub fn negate_y_coordinates<F: Float>(line: &mut na::DVector<F>) {
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

impl<F: Float> Serial<F> {
    /// Construct a new instance
    pub fn new(bounds: Bounds<F>) -> Serial<F> {
        Serial {bounds, stage: Stage::FixChirals}
    }

    /// Calculate distance error value
    pub fn distance_error(bounds: &[DistanceBound<F>], positions: &na::DVector<F>) -> F {
        bounds.iter()
            .map(|bound| bound.error(positions))
            .sum()
    }

    /// Calculate distance gradient vector
    pub fn distance_gradient(bounds: &[DistanceBound<F>], positions: &na::DVector<F>) -> na::DVector<F> {
        let mut gradient: na::DVector<F> = na::DVector::zeros(positions.nrows());

        for bound in bounds.iter() {
            if let Some(contribution) = bound.gradient(positions) {
                contribution.incorporate_into(&mut gradient, bound.indices);
            }
        }

        gradient
    }

    /// Calculate chiral error value
    pub fn chiral_error(bounds: &[Chiral<F>], positions: &na::DVector<F>) -> F {
        bounds.iter()
            .map(|constraint| constraint.error(positions))
            .sum()
    }

    /// Calculate chiral gradient vector
    pub fn chiral_gradient(bounds: &[Chiral<F>], positions: &na::DVector<F>) -> na::DVector<F> {
        let mut gradient: na::DVector<F> = na::DVector::zeros(positions.nrows());

        for constraint in bounds.iter() {
            if let Some(contribution) = constraint.gradient(positions) {
                contribution.incorporate_into(&mut gradient, &constraint.sites);
            }
        }

        gradient
    }

    /// Calculate fourth dimension error value
    pub fn fourth_dimension_error(positions: &na::DVector<F>, stage: &Stage) -> F {
        match stage {
            Stage::FixChirals => <F as num_traits::Zero>::zero(),
            Stage::CompressFourthDimension => {
                let n = positions.len() / DIMENSIONS;
                (0..n).map(|i| num_traits::Float::powi(positions[DIMENSIONS * i + 3], 2))
                    .sum()
            }
        }
    }

    /// Add fourth dimension gradient to existing gradient
    pub fn fourth_dimension_gradient(positions: &na::DVector<F>, mut gradient: na::DVector<F>, stage: &Stage) -> na::DVector<F> {
        match stage {
            Stage::FixChirals => gradient,
            Stage::CompressFourthDimension => {
                let n = positions.len() / DIMENSIONS;
                for i in 0..n {
                    gradient[DIMENSIONS * i + 3] += F::from(2.0).unwrap() * positions[DIMENSIONS * i + 3];
                }
                gradient
            }
        }
    }
}

impl<F: Float> RefinementErrorFunction<F> for Serial<F> {
    fn error(&self, positions: &na::DVector<F>) -> F {
        Self::distance_error(&self.bounds.distances, positions)
            + Self::chiral_error(&self.bounds.chirals, positions)
            + Self::fourth_dimension_error(positions, &self.stage)
    }

    fn gradient(&self, positions: &na::DVector<F>) -> na::DVector<F> {
        let gradient = Self::distance_gradient(&self.bounds.distances, positions) 
            + Self::chiral_gradient(&self.bounds.chirals, positions);
        Self::fourth_dimension_gradient(positions, gradient, &self.stage)
    }

    fn set_stage(&mut self, stage: Stage) {
        self.stage = stage;
    }
}

/// Parallel computation of error function
pub struct Parallel<F: Float> {
    bounds: Bounds<F>,
    stage: Stage
}

impl<F: Float> Parallel<F> {
    /// Construct a new instance
    pub fn new(bounds: Bounds<F>) -> Parallel<F> {
        Parallel {bounds, stage: Stage::FixChirals}
    }

    /// Calculate distance error value
    pub fn distance_error(bounds: &[DistanceBound<F>], positions: &na::DVector<F>) -> F {
        bounds.par_iter()
            .map(|bound| bound.error(positions))
            .sum()
    }

    /// Calculate distance gradient vector
    pub fn distance_gradient(bounds: &[DistanceBound<F>], positions: &na::DVector<F>) -> na::DVector<F> {
        // Calculate gradient contributions for each bound
        let contributions: Vec<Option<DistanceBoundGradient<F>>> = bounds.par_iter()
            .map(|bound| bound.gradient(positions))
            .collect();

        // Serially collect the gradient
        let mut gradient: na::DVector<F> = na::DVector::zeros(positions.nrows());
        for (maybe_contribution, bound) in contributions.into_iter().zip(bounds.iter()) {
            if let Some(contribution) = maybe_contribution {
                contribution.incorporate_into(&mut gradient, bound.indices);
            }
        }
        gradient
    }

    /// Calculate chiral error value
    pub fn chiral_error(bounds: &[Chiral<F>], positions: &na::DVector<F>) -> F {
        bounds.par_iter()
            .map(|constraint| constraint.error(positions))
            .sum()
    }

    /// Calculate chiral gradient vector
    pub fn chiral_gradient(bounds: &[Chiral<F>], positions: &na::DVector<F>) -> na::DVector<F> {
        // Calculate chiral gradient contributions in parallel
        let contributions: Vec<Option<ChiralGradient<F>>> = bounds.par_iter()
            .map(|constraint| constraint.gradient(positions))
            .collect();

        // Serially collect the gradient contributions
        let mut gradient: na::DVector<F> = na::DVector::zeros(positions.nrows());
        for (constraint, maybe_contribution) in bounds.iter().zip(contributions.into_iter()) {
            if let Some(contribution) = maybe_contribution {
                contribution.incorporate_into(&mut gradient, &constraint.sites);
            }
        }
        gradient
    }

    /// Calculate fourth dimension error value
    pub fn fourth_dimension_error(positions: &na::DVector<F>, stage: &Stage) -> F {
        match stage {
            Stage::FixChirals => <F as num_traits::Zero>::zero(),
            Stage::CompressFourthDimension => {
                let n = positions.len() / DIMENSIONS;
                (0..n).into_par_iter()
                    .map(|i| num_traits::Float::powi(positions[DIMENSIONS * i + 3], 2))
                    .sum()
            }
        }
    }

    /// Add fourth dimension gradient to existing gradient
    pub fn fourth_dimension_gradient(positions: &na::DVector<F>, gradient: na::DVector<F>, stage: &Stage) -> na::DVector<F> {
        match stage {
            Stage::FixChirals => gradient,
            Stage::CompressFourthDimension => {
                // Positions could be reshaped for a nicer form
                //
                //  let matrix_grad = gradient.reshape_generic(na::Const<DIMENSIONS>, na::Dyn(n));
                //  matrix_grad.par_column_iter_mut()
                //      .zip(positions.par_column_iter())
                //      .for_each(|(mut grad_v, pos_v)| grad_v.w() += 2.0 * pos_v.w(););

                let n = positions.len() / DIMENSIONS;
                let mut matrix_grad = gradient.reshape_generic(na::Const::<DIMENSIONS>, na::Dyn(n));
                matrix_grad.par_column_iter_mut()
                    .enumerate()
                    .for_each(|(i, mut grad_v)| {grad_v[3] += F::from(2.0).unwrap() * positions[DIMENSIONS * i + 3];});

                matrix_grad.reshape_generic(na::Dyn(n * DIMENSIONS), na::Const::<1>)
            }
        }
    }
}

impl<F: Float> RefinementErrorFunction<F> for Parallel<F> {
    fn error(&self, positions: &na::DVector<F>) -> F {
        Self::distance_error(&self.bounds.distances, positions)
            + Self::chiral_error(&self.bounds.chirals, positions)
            + Self::fourth_dimension_error(positions, &self.stage)
    }

    fn gradient(&self, positions: &na::DVector<F>) -> na::DVector<F> {
        let gradient = Self::distance_gradient(&self.bounds.distances, positions) 
            + Self::chiral_gradient(&self.bounds.chirals, positions);
        Self::fourth_dimension_gradient(positions, gradient, &self.stage)
    }

    fn set_stage(&mut self, stage: Stage) {
        self.stage = stage;
    }
}

/// Interface between various error function implementations and argmin
pub trait RefinementErrorFunction<F> {
    /// Calculate the error function value
    fn error(&self, positions: &na::DVector<F>) -> F;
    /// Calculate the error function gradient
    fn gradient(&self, positions: &na::DVector<F>) -> na::DVector<F>;
    /// Set the stage of refinement
    fn set_stage(&mut self, stage: Stage);
}

impl<F: Float> argmin::core::CostFunction for &dyn RefinementErrorFunction<F> {
    type Param = na::DVector<F>;
    type Output = F;

    fn cost(&self, param: &Self::Param) -> Result<Self::Output, argmin::core::Error> {
        Ok(self.error(param))
    }
}

/// Calculate error function gradient for argmin
impl<F: Float> argmin::core::Gradient for &dyn RefinementErrorFunction<F> {
    type Param = <Self as argmin::core::CostFunction>::Param;
    type Gradient = na::DVector<F>;

    fn gradient(&self, param: &Self::Param) -> Result<Self::Gradient, argmin::core::Error> {
        Ok((*self).gradient(param))
    }
}

/// Result data of a refinement
pub struct Refinement<F: Float> {
    /// Refined coordinates
    pub coords: na::Matrix3xX<F>,
    /// Number of minimization steps
    pub steps: usize
}

/// Possible errors in refinement
#[derive(Error, Debug, PartialEq, Eq)]
pub enum RefinementError {
    /// Function minimization optimizer ran into an issue
    ///
    /// Either the error function value and gradient methods are not correctly implemented 
    /// or there is an issue with the chosen argmin optimizer and/or line search method.
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

/// Refine a set of coordinates with some error function implementation
pub fn refine<F: Float>(mut problem: impl RefinementErrorFunction<F>, positions: na::Matrix4xX<F>) -> Result<Refinement<F>, RefinementError> 
    where for<'a> &'a dyn RefinementErrorFunction<F>: argmin::core::CostFunction<Output=F, Param=na::DVector<F>>
{
    const LBFGS_MEMORY: usize = 32;
    let tolerance_grad: F = F::from_f32(1e-6).expect("Representable constant");

    let n = positions.len();
    let linear_positions = positions.reshape_generic(na::Dyn(n), na::Const::<1>);
    // TODO invert y if beneficial to chirals

    let linesearch = MoreThuenteLineSearch::new();

    problem.set_stage(Stage::FixChirals);
    let solver = LBFGS::new(linesearch.clone(), LBFGS_MEMORY)
        .with_tolerance_grad(tolerance_grad)
        .expect("LBFGS constructible");
    let mut result = Executor::new(&problem as &dyn RefinementErrorFunction<F>, solver)
        .configure(|state| state.param(linear_positions))
        .run()
        .unwrap();
    check_termination_status(&result.state)?;
    let uncompressed_positions = result.state.take_best_param()
        .expect("Successful termination implies optimal parameters present");
    let relaxation_steps = result.state.iter;

    problem.set_stage(Stage::CompressFourthDimension);
    let solver = LBFGS::new(linesearch, LBFGS_MEMORY)
        .with_tolerance_grad(tolerance_grad)
        .expect("LBFGS constructible");
    let mut result = Executor::new(&problem as &dyn RefinementErrorFunction<F>, solver)
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

/// Refinement functions on the gpu
#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(test)]
mod tests {
    use crate::dg::{DistanceMatrix, MetricMatrix, MetrizationPartiality};
    use crate::dg::refinement::*;
    use crate::shapes::TETRAHEDRON;

    extern crate nalgebra as na;

    /// Central difference numerical gradient
    fn numerical_gradient<F: Float>(cost: &dyn Fn(&na::DVector<F>) -> F, args: &na::DVector<F>) -> na::DVector<F> {
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
            epsilon=1e-6
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
            epsilon=1e-6
        );
    }

    #[test]
    fn chiral_bound_gradient() {
        let shape = &TETRAHEDRON;
        let bounds = crate::dg::modeling::solitary_shape::shape_into_bounds(shape);
        let distances = DistanceMatrix::try_from_distance_bounds(bounds, MetrizationPartiality::Complete).expect("Successful metrization");
        let metric = MetricMatrix::from(distances);
        let coords = metric.embed();
        let n = coords.len();
        let mut linear_coords = coords.reshape_generic(na::Dyn(n), na::Const::<1>);
        let chirals: Vec<Chiral<f64>> = shape.find_tetrahedra().into_iter()
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
