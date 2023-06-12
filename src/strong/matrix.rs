extern crate nalgebra as na;
type Matrix3N = na::Matrix3xX<f64>;

use num_traits::ToPrimitive;

use crate::strong::NewTypeIndex;
use crate::shapes::similarity::apply_permutation;
use crate::strong::bijection::Bijection;
use crate::quaternions;

use std::collections::HashMap;
use std::marker::PhantomData;

pub struct AsNewTypeIndexedMatrix<'a, I> where I: NewTypeIndex {
    pub matrix: &'a Matrix3N,
    index_type: PhantomData<I>
}

impl<'a, I> AsNewTypeIndexedMatrix<'a, I> where I: NewTypeIndex {
    pub fn new(matrix: &'a Matrix3N) -> AsNewTypeIndexedMatrix<'a, I> {
        AsNewTypeIndexedMatrix {matrix, index_type: PhantomData}
    }

    /// Access a column of the matrix
    pub fn column(&self, index: I) -> na::VectorView3<'a, f64> {
        self.matrix.column(index.get().to_usize().expect("Conversion failure"))
    }

    pub fn quaternion_fit_with_rotor(&self, rotor: AsNewTypeIndexedMatrix<I>) -> quaternions::Fit {
        quaternions::fit(self.matrix, rotor.matrix)
    }

    pub fn quaternion_fit_with_map<J>(&self, rotor: AsNewTypeIndexedMatrix<J>, map: &HashMap<I, J>) -> quaternions::Fit where J: NewTypeIndex {
        assert!(self.matrix.column_mean().norm_squared() < 1e-8);
        assert!(rotor.matrix.column_mean().norm_squared() < 1e-8);
        
        let mut a = nalgebra::Matrix4::<f64>::zeros();
        for (stator_i, rotor_i) in map {
            let stator_col = self.column(*stator_i);
            let rotor_col = rotor.column(*rotor_i);
            a += quaternions::quaternion_pair_contribution(&stator_col, &rotor_col);
        }

        quaternions::quaternion_decomposition(a)
    }

    /// Permute the matrix into a different index space
    pub fn apply_bijection<J>(&self, bijection: &Bijection<I, J>) -> StrongPoints<J> where J: NewTypeIndex {
        StrongPoints::new(apply_permutation(self.matrix, &bijection.permutation))
    }
}

pub struct StrongPoints<I> where I: NewTypeIndex {
    pub matrix: Matrix3N,
    index_type: PhantomData<I>
}

impl<I> StrongPoints<I> where I: NewTypeIndex {
    fn raise(&self) -> AsNewTypeIndexedMatrix<I> {
        AsNewTypeIndexedMatrix::new(&self.matrix)
    }

    pub fn new(matrix: Matrix3N) -> StrongPoints<I> {
        StrongPoints {matrix, index_type: PhantomData}
    }

    pub fn column(&self, index: I) -> na::VectorView3<f64> {
        self.raise().column(index)
    }

    pub fn quaternion_fit_with_rotor(&self, rotor: &StrongPoints<I>) -> quaternions::Fit {
        self.raise().quaternion_fit_with_rotor(rotor.raise())
    }

    pub fn quaternion_fit_with_map<J>(&self, rotor: &StrongPoints<J>, map: &HashMap<I, J>) -> quaternions::Fit where J: NewTypeIndex {
        self.raise().quaternion_fit_with_map(rotor.raise(), map)
    }

    /// Apply a bijection
    pub fn apply_bijection<J>(&self, bijection: &Bijection<I, J>) -> StrongPoints<J> where J: NewTypeIndex {
        self.raise().apply_bijection(bijection)
    }
}


