use num_traits::ToPrimitive;
use std::collections::HashSet;
use std::marker::PhantomData;
use thiserror::Error;
use crate::strong::Index;
use crate::strong::bijection::IndexBijection;
use crate::permutation::PermutationError;

use std::convert::TryFrom;

/// Struct representing a surjection between index spaces
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct IndexSurjection<T: Index, U: Index> {
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

impl<T: Index, U: Index + Ord + std::hash::Hash> TryFrom<Vec<U>> for IndexSurjection<T, U> {
    type Error = SurjectionError;

    fn try_from(sigma: Vec<U>) -> Result<Self, Self::Error> {
        // The maximum item in value must equal the number of unique items minus one
        let max: U = *sigma.iter().max().ok_or(SurjectionError::EmptyCodomain)?;
        if max.get().to_usize().unwrap() != HashSet::<&U>::from_iter(sigma.iter()).len() - 1 {
            return Err(SurjectionError::NonMinimalCodomain);
        }

        Ok(IndexSurjection {sigma, key_type: PhantomData})
    }
}

impl<T: Index, U: Index + std::convert::From<usize>> From<IndexBijection<T, U>> for IndexSurjection<T, U> 
{
    fn from(bijection: IndexBijection<T, U>)  -> IndexSurjection<T, U> {
        let sigma = Vec::from_iter(bijection.permutation.iter().map(|(_, &u)| U::from(u)));
        IndexSurjection {sigma, key_type: PhantomData}
    }
}

impl<T: Index, U: Index + PartialOrd> IndexSurjection<T, U> {
    /// Identity surjection (also a bijection)
    pub fn identity(size: usize) -> IndexSurjection<T, U> where U: std::convert::From<usize> {
        IndexSurjection::from(IndexBijection::<T, U>::identity(size))
    }

    /// The input domain size of the surjection
    pub fn domain_size(&self) -> usize {
        self.sigma.len()
    }

    /// Access the function value for an input, if within domain
    pub fn get(&self, key: &T) -> Option<U> {
        let index = key.get().to_usize()?;
        self.sigma.get(index).copied()
    }
}

impl<T: Index, U: Index> AsRef<[U]> for IndexSurjection<T, U> {
    fn as_ref(&self) -> &[U] {
        self.sigma.as_slice()
    }
}

impl<T: Index, U: Index> AsMut<[U]> for IndexSurjection<T, U> {
    fn as_mut(&mut self) -> &mut [U] {
        self.sigma.as_mut_slice()
    }
}

/// Trait indicating a type can be surjected
pub trait Surjectable<U: Index + PartialOrd> {
    /// Key/Input type for the surjection
    type Key: Index;
    /// Result of the surjection operation
    type Output;

    /// Perform the surjection
    fn surject(&self, surjection: &IndexSurjection<Self::Key, U>) -> Result<Self::Output, SurjectionError>;
}

impl<T: Index, U: Index, V: Index + Ord + std::hash::Hash> Surjectable<V> for IndexBijection<T, U> 

{
    type Key = U;
    type Output = IndexSurjection<T, V>;

    fn surject(&self, surjection: &IndexSurjection<Self::Key, V>) -> Result<Self::Output, SurjectionError> {
        let sigma = self.permutation.inverse().apply(surjection.sigma.clone()).map_err(|e| {
            assert_eq!(e, PermutationError::LengthMismatch);
            SurjectionError::LengthMismatch
        })?;
        IndexSurjection::try_from(sigma)
    }
}
