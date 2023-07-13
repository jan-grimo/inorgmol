extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use num_traits::ToPrimitive;

use crate::strong::Index;
use crate::shapes::similarity::apply_permutation;
use crate::strong::bijection::Bijection;
use crate::quaternions;

use std::collections::HashMap;
use std::marker::PhantomData;

/// Matrix indexed by a new type
pub struct AsPositions<'a, I> where I: Index {
    matrix: &'a Matrix3N,
    index_type: PhantomData<I>
}

impl<'a, I> AsPositions<'a, I> where I: Index {
    /// Wrap a matrix
    pub fn new(matrix: &'a Matrix3N) -> AsPositions<'a, I> {
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
    pub fn quaternion_fit_with_map<J>(&self, rotor: AsPositions<J>, map: &HashMap<I, J>) -> quaternions::Fit where J: Index {
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

    /// Permute the matrix into a different index space
    pub fn apply_bijection<J>(&self, bijection: &Bijection<I, J>) -> Positions<J> where J: Index {
        Positions::new(apply_permutation(self.matrix, &bijection.permutation))
    }
}

/// Owned positions matrix indexed by a new type
pub struct Positions<I> where I: Index {
    /// The underlying matrix
    pub matrix: Matrix3N,
    index_type: PhantomData<I>
}

impl<I> Positions<I> where I: Index {
    fn raise(&self) -> AsPositions<I> {
        AsPositions::new(&self.matrix)
    }

    /// Wrap a matrix
    pub fn new(matrix: Matrix3N) -> Positions<I> {
        Positions {matrix, index_type: PhantomData}
    }

    /// Access the position of a particular point
    pub fn point(&self, index: I) -> na::VectorView3<f64> {
        self.raise().point(index)
    }

    /// Quaternion fit a matrix in the same index space
    pub fn quaternion_fit_with_rotor(&self, rotor: &Positions<I>) -> quaternions::Fit {
        self.raise().quaternion_fit_rotor(rotor.raise())
    }

    /// Quaternion fit a matrix (in a possibly different index space) with an index mapping
    pub fn quaternion_fit_with_map<J>(&self, rotor: &Positions<J>, map: &HashMap<I, J>) -> quaternions::Fit where J: Index {
        self.raise().quaternion_fit_with_map(rotor.raise(), map)
    }

    /// Apply a bijection
    pub fn apply_bijection<J>(&self, bijection: &Bijection<I, J>) -> Positions<J> where J: Index {
        self.raise().apply_bijection(bijection)
    }

    /// Extract the underlying matrix
    pub fn take_matrix(self) -> Matrix3N {
        self.matrix
    }
}


