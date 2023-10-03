use num_traits::ToPrimitive;
use std::collections::HashSet;
use std::marker::PhantomData;
use thiserror::Error;
use crate::strong::Index;
use crate::strong::bijection::Bijection;
use crate::permutation::{slice_next, slice_prev, slice_permutation_index, PermutationError};

use std::convert::TryFrom;

/// Struct representing a surjection between index spaces
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct Surjection<T: Index, U: Index> {
    /// One-line representation of the surjective function with minimal domain
    pub sigma: Vec<U>,
    key_type: PhantomData<T>
}

/// Errors arising in use of surjections
#[derive(Error, Debug, PartialEq, Eq)]
pub enum SurjectionError {
    /// Codomain of surjection is not a natural numbers sequence including zero
    #[error("Codomain is not minimal")]
    NonMinimalCodomain,
    /// The codomain of the surjection is empty
    #[error("Empty codomain")]
    EmptyCodomain,
    /// The length of the surjection mismatches with an argument
    #[error("Length mismatch")]
    LengthMismatch
}

impl<T: Index, U: Index + Ord + std::hash::Hash> TryFrom<Vec<U>> for Surjection<T, U> {
    type Error = SurjectionError;

    fn try_from(sigma: Vec<U>) -> Result<Self, Self::Error> {
        // The maximum item in value must equal the number of unique items minus one
        let max: U = *sigma.iter().max().ok_or(SurjectionError::EmptyCodomain)?;
        if max.get().to_usize().unwrap() != HashSet::<&U>::from_iter(sigma.iter()).len() - 1 {
            return Err(SurjectionError::NonMinimalCodomain);
        }

        Ok(Surjection {sigma, key_type: PhantomData})
    }
}

impl<T: Index, U: Index + std::convert::From<usize>> From<Bijection<T, U>> for Surjection<T, U> 
{
    fn from(bijection: Bijection<T, U>)  -> Surjection<T, U> {
        let sigma = Vec::from_iter(bijection.permutation.iter().map(|(_, &u)| U::from(u)));
        Surjection {sigma, key_type: PhantomData}
    }
}

impl<T: Index, U: Index + PartialOrd> Surjection<T, U> {
    /// Identity surjection (also a bijection)
    pub fn identity(size: usize) -> Surjection<T, U> where U: std::convert::From<usize> {
        Surjection::from(Bijection::<T, U>::identity(size))
    }

    /// The input domain size of the surjection
    pub fn domain_size(&self) -> usize {
        self.sigma.len()
    }

    /// Access the function value for an input, if within domain
    pub fn get(&self, key: &T) -> Option<U> {
        let index = key.get().to_usize()?;
        return self.sigma.get(index).copied();
    }

    /// Yield the index of permutation of the surjection
    pub fn index(&self) -> usize {
        slice_permutation_index(&self.sigma)
    }

    /// Transform into the next permutation within the partial order of its multiset
    pub fn next_permutation(&mut self) -> bool {
        slice_next(self.sigma.as_mut_slice())
    }

    /// Transform into the previous permutation within the partial order of its multiset
    pub fn prev_permutation(&mut self) -> bool {
        slice_prev(self.sigma.as_mut_slice())
    }

    /// Iterate through all permutations of the surjection
    pub fn iter_permutations(&self) -> SurjectionIterator<T, U> {
        SurjectionIterator::new(self.clone())
    }
}

/// Iterator for permutations of a surjection
#[derive(Clone)]
pub struct SurjectionIterator<T: Index, U: Index + PartialOrd> {
    surjection: Surjection<T, U>,
    increment: bool
}

impl<T: Index, U: Index + PartialOrd> SurjectionIterator<T, U> {
    fn new(surjection: Surjection<T, U>) -> SurjectionIterator<T, U> {
        SurjectionIterator {surjection, increment: false}
    }
}

impl<T: Index, U: Index + PartialOrd> Iterator for SurjectionIterator<T, U> {
    type Item = Surjection<T, U>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.increment && !self.surjection.next_permutation() {
            return None;
        }

        self.increment = true;
        Some(self.surjection.clone())
    }
}

/// Trait indicating a type can be surjected
pub trait Surjectable<U: Index + PartialOrd> {
    /// Key/Input type for the surjection
    type Key: Index;
    /// Result of the surjection operation
    type Output;

    /// Perform the surjection
    fn surject(&self, surjection: &Surjection<Self::Key, U>) -> Result<Self::Output, SurjectionError>;
}

impl<T: Index, U: Index, V: Index + Ord + std::hash::Hash> Surjectable<V> for Bijection<T, U> 

{
    type Key = U;
    type Output = Surjection<T, V>;

    fn surject(&self, surjection: &Surjection<Self::Key, V>) -> Result<Self::Output, SurjectionError> {
        let sigma = self.permutation.inverse().apply(surjection.sigma.clone()).map_err(|e| {
            assert_eq!(e, PermutationError::LengthMismatch);
            SurjectionError::LengthMismatch
        })?;
        Surjection::try_from(sigma)
    }
}
