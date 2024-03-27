extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use num_traits::ToPrimitive;

use crate::strong::Index;
use crate::strong::bijection::{IndexBijection, Bijectable};
use crate::permutation::{PermutationError, Permutatable};
use crate::quaternions;

use std::collections::HashMap;
use std::marker::PhantomData;

/// Matrix indexed by a new type
pub struct AsPositions<'a, I: Index> {
    matrix: &'a Matrix3N,
    index_type: PhantomData<I>
}

impl<'a, I: Index> AsPositions<'a, I> {
    /// Wrap a matrix
    pub fn wrap(matrix: &'a Matrix3N) -> AsPositions<'a, I> {
        AsPositions {matrix, index_type: PhantomData}
    }

    /// Access the position of a particular point
    pub fn point(&self, index: I) -> na::VectorView3<'a, f64> {
        self.matrix.column(index.get().to_usize().expect("Conversion failure"))
    }

    /// Quaternion fit a matrix in the same index space
    pub fn quaternion_fit_rotor(&self, rotor: AsPositions<I>) -> quaternions::Fit {
        quaternions::fit(self.matrix, rotor.matrix)
    }

    /// Quaternion fit a matrix (in a possibly different index space) with an index mapping
    pub fn quaternion_fit_map<J: Index>(&self, rotor: AsPositions<J>, map: &HashMap<I, J>) -> quaternions::Fit {
        assert!(self.matrix.column_mean().norm_squared() < 1e-8);
        assert!(rotor.matrix.column_mean().norm_squared() < 1e-8);
        
        let mut a = nalgebra::Matrix4::<f64>::zeros();
        for (stator_i, rotor_i) in map {
            let stator_col = self.point(*stator_i);
            let rotor_col = rotor.point(*rotor_i);
            a += quaternions::quaternion_pair_contribution(&stator_col, &rotor_col);
        }

        quaternions::quaternion_decomposition(a)
    }
}

impl<'a, I: Index, U: Index> Bijectable<U> for AsPositions<'a, I> {
    type T = I;
    type Output = Positions<U>;

    fn biject(&self, bijection: &IndexBijection<I, U>) -> Result<<Self as Bijectable<U>>::Output, PermutationError> {
        if bijection.set_size() != self.matrix.ncols() {
            return Err(PermutationError::LengthMismatch);
        }

        let matrix = self.matrix.permute(&bijection.permutation)?;
        Ok(Positions::wrap(matrix))
    }
}

/// Owned positions matrix indexed by a new type
#[derive(Clone)]
pub struct Positions<I: Index> {
    /// The underlying matrix
    pub matrix: Matrix3N,
    index_type: PhantomData<I>
}

impl<I: Index> Positions<I> {
    fn raise(&self) -> AsPositions<I> {
        AsPositions::wrap(&self.matrix)
    }

    /// Wrap a matrix
    pub fn wrap(matrix: Matrix3N) -> Positions<I> {
        Positions {matrix, index_type: PhantomData}
    }

    /// Access the position of a particular point
    pub fn point(&self, index: I) -> na::VectorView3<f64> {
        self.raise().point(index)
    }

    /// Quaternion fit a matrix in the same index space
    pub fn quaternion_fit_rotor(&self, rotor: &Positions<I>) -> quaternions::Fit {
        self.raise().quaternion_fit_rotor(rotor.raise())
    }

    /// Quaternion fit a matrix (in a possibly different index space) with an index mapping
    pub fn quaternion_fit_map<J: Index>(&self, rotor: &Positions<J>, map: &HashMap<I, J>) -> quaternions::Fit {
        self.raise().quaternion_fit_map(rotor.raise(), map)
    }

    /// Extract the underlying matrix
    pub fn take_matrix(self) -> Matrix3N {
        self.matrix
    }
}

impl<I: Index, U: Index> Bijectable<U> for Positions<I> {
    type T = I;
    type Output = Positions<U>;

    fn biject(&self, bijection: &IndexBijection<I, U>) -> Result<<Self as Bijectable<U>>::Output, PermutationError> {
        self.raise().biject(bijection)
    }
}
