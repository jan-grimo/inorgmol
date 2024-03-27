use std::marker::PhantomData;
use num_traits::ToPrimitive;
use num_traits::FromPrimitive;
use delegate::delegate;

use crate::permutation::{Permutation, PermutationError};
use crate::strong::Index;
use std::convert::TryFrom;

/// Struct representing a bijection between index spaces
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct IndexBijection<T: Index, U: Index> {
    /// Underlying weakly index-typed Permutation 
    pub permutation: Permutation,
    key_type: PhantomData<T>,
    value_type: PhantomData<U>
}

impl<T: Index, U: Index> IndexBijection<T, U> {
    /// Initialize by wrapping a Permutation
    pub fn new(p: Permutation) -> IndexBijection<T, U> {
        IndexBijection {permutation: p, key_type: PhantomData, value_type: PhantomData}
    }

    /// Generates a random bijection (identity inclusive)
    pub fn new_random(n: usize) -> IndexBijection<T, U> {
        IndexBijection::new(Permutation::new_random(n))
    }

    /// Initialize the `i`-th bijection by lexicographic order of size `n`
    ///
    /// Initializes a particular permutation identified by its position
    /// (zero-indexed) in the lexicographic order `i` of all permutations of
    /// size `n`.
    ///
    /// If `i` is larger than the final position of all permutations of that
    /// size, i.e. $ i >= n! $, `None` is returned.
    pub fn try_from_index(n: usize, i: usize) -> Option<IndexBijection<T, U>> {
        Permutation::try_from_index(n, i).map(IndexBijection::new)
    }

    /// Initialize an identity bijection of specific size
    ///
    /// The identity bijection maps each number onto itself. It has index
    /// zero within the lexicographical order of bijections.
    pub fn identity(n: usize) -> IndexBijection<T, U> {
        IndexBijection::new(Permutation::identity(n))
    }

    /// Invert the bijection
    pub fn inverse(&self) -> IndexBijection<U, T> {
        IndexBijection::new(self.permutation.inverse())
    }

    /// Apply the map to a key and find its corresponding value
    pub fn get(&self, key: &T) -> Option<U> {
        let index = key.get().to_usize()?;
        if index >= self.permutation.set_size() {
            return None;
        }

        let value = self.permutation[index];
        Some(U::from(<U::Type as FromPrimitive>::from_usize(value)?))
    }

    /// Find the key to a corresponding value
    pub fn inverse_of(&self, value: &U) -> Option<T> {
        let inverse = self.permutation.inverse_of(value.get().to_usize()?)?;
        let key = T::from(<T::Type as FromPrimitive>::from_usize(inverse)?);
        Some(key)
    }

    /// Compose the bijection with another
    pub fn compose<V: Index>(&self, other: &IndexBijection<U, V>) 
    -> Result<IndexBijection<T, V>, PermutationError> 
    {
        let p = self.permutation.compose(&other.permutation)?;
        Ok(IndexBijection::new(p))
    }

    delegate! {
        to self.permutation {
            /// Determine the index of a bijection in its lexicographic order
            pub fn index(&self) -> usize;
            /// Transform into the next permutation within the partial order of its set
            pub fn next_permutation(&mut self) -> bool;
            /// Transform into the previous permutation within its set's partial order
            pub fn prev_permutation(&mut self) -> bool;
            /// Number of elements being bijected
            pub fn set_size(&self) -> usize;
        }
    }

    /// Number of possible bijections
    pub fn group_order(n: usize) -> usize {
        Permutation::group_order(n)
    }

    /// Fixed points are values that the bijection maps onto itself
    pub fn is_fixed_point(&self, key: T) -> bool {
        self.permutation.is_fixed_point(key.get().to_usize().unwrap())
    }
}

impl<T: Index, U: Index> std::fmt::Display for IndexBijection<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.permutation)
    }
}

impl<T: Index, U: Index> TryFrom<Vec<U>> for IndexBijection<T, U> {
    type Error = PermutationError;

    fn try_from(strong_vec: Vec<U>) -> Result<IndexBijection<T, U>, Self::Error> {
        let weak_vec: Vec<usize> = strong_vec.into_iter()
            .map(|v| v.get().to_usize().unwrap())
            .collect();
        Permutation::try_from(weak_vec).map(|p| IndexBijection::new(p))
    }
}

/// Iterator adaptor for iterating through all bijections of a set size
///
/// See [`bijections`]
pub struct BijectionsIterator<T: Index, U: Index> {
    bijection: IndexBijection<T, U>,
    increment: bool
}

impl<T: Index, U: Index> BijectionsIterator<T, U> {
    fn new(bijection: IndexBijection<T, U>) -> BijectionsIterator<T, U> {
        BijectionsIterator {bijection, increment: false}
    }
}

impl<T: Index, U: Index> Iterator for BijectionsIterator<T, U> {
    type Item = IndexBijection<T, U>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.increment && !self.bijection.permutation.next_permutation() {
            return None;
        }

        self.increment = true;
        Some(self.bijection.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let p = &self.bijection.permutation;
        let remaining = Permutation::group_order(p.set_size()) - p.index();
        (remaining, Some(remaining))
    }
}

/// Yields bijections in increasing lexicographic order
pub fn bijections<T: Index, U: Index>(n: usize) -> BijectionsIterator<T, U> {
    BijectionsIterator::<T, U>::new(IndexBijection::identity(n))
}

/// Trait indicating a type can be bijected
pub trait Bijectable<U: Index> {
    /// Key/Input type for the bijection
    type T: Index;
    /// Result of the bijection
    type Output;

    /// Perform the bijection
    fn biject(&self, bijection: &IndexBijection<Self::T, U>) -> Result<Self::Output, PermutationError>;
}

impl<T: Index, U: Index> Bijectable<U> for Vec<T> {
    type T = T;
    type Output = Vec<U>;

    fn biject(&self, bijection: &IndexBijection<T, U>) -> Result<Self::Output, PermutationError> {
        let vec = Vec::from_iter(self.iter().filter_map(|item| bijection.get(item)));
        if vec.len() == self.len() {
            Ok(vec)
        } else {
            Err(PermutationError::LengthMismatch)
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::permutation::Permutation;
    use crate::strong::bijection::{IndexBijection, Bijectable};
    use crate::strong::IndexBase;

    #[derive(IndexBase, Debug, Copy, Clone, PartialEq)]
    struct Foo(usize);

    #[derive(IndexBase, Debug, Copy, Clone, PartialEq)]
    struct Bar(usize);

    #[derive(IndexBase, Debug, Copy, Clone, PartialEq)]
    struct Baz(usize);

    #[test]
    fn basics() {
        let f1 = IndexBijection::<Foo, Bar>::try_from_index(4, 3).expect("Valid index");
        let f1_at_two = Bar::from(f1.permutation[2]);
        assert_eq!(f1.get(&Foo(2)), Some(f1_at_two));
        assert_eq!(f1.inverse_of(&f1_at_two), Some(Foo::from(2)));

        let f2 = IndexBijection::<Bar, Baz>::try_from_index(4, 3).expect("Valid index");
        let f3 = f1.compose(&f2).unwrap();
        assert_eq!(f3.permutation, f1.permutation.compose(&f2.permutation).unwrap());

        // expect compile error
        // let f4 = f1.compose(&f3);

        let p1 = IndexBijection::<Bar, Bar>::new(Permutation::identity(4));
        let f4 = f1.compose(&p1).unwrap();
        assert_eq!(f1.permutation, f4.permutation);
    }

    #[test]
    fn biject() {
        let foos = vec![Foo(4), Foo(2)];
        let bijection = IndexBijection::<Foo, Bar>::try_from_index(5, 5).expect("Valid index");
        let bars = foos.biject(&bijection);
        assert!(bars.is_ok());
        let refoo = bars.unwrap().biject(&bijection.inverse());
        assert!(refoo == Ok(foos));

        // expect compile error
        // let failing = Bijection::<Bar, Baz>::try_from_index(1, 1).expect("Valid index");
        // let bazzes = foos.biject(&failing);
    }
}
