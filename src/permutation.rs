/// TODO
/// - from_index could fail with indices higher than count()

use std::ops::Index;
use core::convert::TryFrom;
use num_traits::int::PrimInt;
use itertools::Itertools;
use thiserror::Error;

/// Slice-level permutation incrementation
pub fn slice_next<T: PartialOrd>(slice: &mut [T]) -> bool {
    let n = slice.len();

    if n == 0 {
        return false;
    }

    let mut i = n - 1;
    let mut j;
    let mut k;

    loop {
        j = i;

        if i != 0 {
            i -= 1;

            if slice[i] < slice[j] {
                k = n - 1;
                loop {
                    if k == 0 || slice[i] < slice[k] {
                        break
                    }
                    k -= 1;
                }

                slice.swap(i, k);
                slice[j..n].reverse();
                break true
            }
        } else {
            slice.reverse();
            break false
        }
    }
}

/// Slice-level permutation decrementation
pub fn slice_prev<T: PartialOrd>(slice: &mut [T]) -> bool {
    let n = slice.len();

    if n == 0 {
        return false;
    }

    let mut i = n - 1;
    let mut j;
    let mut k;

    loop {
        j = i;

        if i != 0 {
            i -= 1;

            if slice[j] < slice[i] {
                k = n - 1;

                loop {
                    if k == 0 || slice[k] < slice[i] {
                        break
                    }
                    k -= 1;
                }

                slice.swap(i, k);
                slice[j..n].reverse();
                break true
            }
        } else {
            slice.reverse();
            break false
        }
    }
}

pub fn slice_apply<T: PrimInt, U: Copy>(permutation: &[T], values: &[U]) -> Option<Vec<U>> {
    let n = permutation.len();
    if n != values.len() {
        return None;
    } 

    let mut permuted = values.to_vec();
    for i in 0..n {
        permuted[permutation[i].to_usize().unwrap()] = values[i];
    }
    Some(permuted)
}

#[derive(PartialEq, Eq, PartialOrd, Clone, Debug, Hash)]
pub struct Permutation {
    // One-line representation
    pub sigma: Vec<u8>
}

impl std::fmt::Display for Permutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{}}}", self.sigma.iter().format(", "))
    }
}

#[derive(Error, Debug, PartialEq, Eq)]
pub enum PermutationError {
    #[error("Mismatched lengths between permutation and argument")]
    LengthMismatch,
    #[error("More than u8::MAX elements")]
    NotRepresentable,
    #[error("Invalid elements of one-line representation set")]
    InvalidSetElements
}

impl Permutation {
    /// Initialize an identity permutation of specific size
    ///
    /// The identity permutation has index zero within the lexicographical order 
    /// of permutations.
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation::identity(3).sigma, vec![0, 1, 2])
    /// ```
    pub fn identity(n: usize) -> Permutation {
        Permutation {sigma: (0..n as u8).collect()}
    }

    /// Initialize the i-th permutation by lexicographic order of size n
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation::from_index(3, 0).sigma, vec![0, 1, 2]);
    /// assert_eq!(Permutation::from_index(3, 1).sigma, vec![0, 2, 1]);
    /// assert_eq!(Permutation::from_index(3, 2).sigma, vec![1, 0, 2]);
    /// ```
    pub fn from_index(n: usize, mut i: usize) -> Permutation {
        let mut factorials = Vec::<usize>::with_capacity(n);
        factorials.push(1);
        for k in 1..n {
            factorials.push(factorials.last().unwrap() * k);
        }

        let mut sigma = Vec::with_capacity(n);

        for k in 0..n {
            let fac = factorials[n - 1 - k];
            sigma.push((i / fac) as u8);
            i %= fac;
        }

        for k in (1..n).rev() {
            for j in (0..k).rev() {
                if sigma[j] <= sigma[k] {
                    sigma[k] += 1;
                }
            }
        }

        Permutation {sigma}
    }

    /// Find a permutation ordering a container's elements
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let container = vec![-3, 2, 0];
    /// let p = Permutation::ordering(&container);
    /// assert_eq!(p, Permutation::from_index(3, 1));
    /// assert_eq!(p.apply(&container), Ok(vec![-3, 0, 2]));
    /// ```
    pub fn ordering<T: Ord>(container: &Vec<T>) -> Permutation {
        let mut p = Permutation::identity(container.len());
        p.sigma.sort_by(|i, j| container[*i as usize].cmp(&container[*j as usize]));
        p
    }

    /// Determine the index of a permutation in its lexicographic order
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation {sigma: vec![0, 1, 2]}.index(), 0);
    /// assert_eq!(Permutation {sigma: vec![0, 2, 1]}.index(), 1);
    /// for i in 0..10 {
    ///     assert_eq!(Permutation::from_index(6, i).index(), i);
    /// }
    /// ```
    pub fn index(&self) -> usize {
        let n = self.sigma.len();

        if n == 0 {
            return 0;
        }

        let mut index = 0;
        let mut position = 2;
        let mut factor = 1;

        for p in (0..(n - 1)).rev() {
            let is_smaller = |q| { (self.sigma[q] < self.sigma[p]) as usize};
            let larger_successors: usize = ((p + 1)..n).map(is_smaller).sum();
            index += larger_successors * factor;
            factor *= position;
            position += 1;
        }

        index
    }

    /// Number of elements being permuted
    pub fn set_size(&self) -> usize {
        self.sigma.len()
    }

    /// Determine the number of possible permutations, also known as the order of the symmetric
    /// group spanned by the Permutation
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation::group_order(3), 6);
    /// ```
    pub fn group_order(n: usize) -> usize {
        (1..=n).product()
    }

    /// Transform into the next permutation within the partial order of its set
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let mut permutation = Permutation::from_index(6, 0);
    /// for i in 1..15 {
    ///     assert_eq!(permutation.next_permutation(), true);
    ///     assert_eq!(permutation, Permutation::from_index(6, i));
    /// }
    /// ```
    pub fn next_permutation(&mut self) -> bool {
        slice_next(self.sigma.as_mut_slice())
    }

    /// Transform into the previous permutation within its set's partial order
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let mut permutation = Permutation::from_index(6, 15);
    /// for i in 14..1 {
    ///     assert_eq!(permutation.prev_permutation(), true);
    ///     assert_eq!(permutation, Permutation::from_index(6, i));
    /// }
    /// ```
    pub fn prev_permutation(&mut self) -> bool {
        slice_prev(self.sigma.as_mut_slice())
    }

    /// Generate the inverse permutation to the current permutation
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::result::Result;
    /// # fn main() -> Result<(), PermutationError> {
    /// let permutation = Permutation::from_index(3, 1);
    /// assert_eq!(permutation.inverse().compose(&permutation)?, Permutation::identity(3));
    /// assert_eq!(permutation.compose(&permutation.inverse())?, Permutation::identity(3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn inverse(&self) -> Permutation {
        let n = self.sigma.len();
        let mut inverse = Permutation::identity(n);
        for i in 0..n {
            inverse.sigma[self.sigma[i] as usize] = i as u8;
        }
        inverse
    }

    /// Apply the permutation to a vector
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// let p = Permutation::try_from([1, 2, 0])?;
    /// let v = vec!["I", "am", "Yoda"];
    /// assert_eq!(p.apply(&v), Ok(vec!["Yoda", "I", "am"]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply<T: Copy>(&self, other: &Vec<T>) -> Result<Vec<T>, PermutationError> {
        slice_apply(self.sigma.as_slice(), other.as_slice()).ok_or(PermutationError::LengthMismatch)
    }

    /// Compose two permutations into a new permutation
    ///
    /// The resulting permutation applies `self` first, then `other`. Note that 
    /// permutation composition is not commutative.
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// // Usual case of non-commutative composition
    /// let p = Permutation::try_from([1, 2, 0])?;
    /// let q = Permutation::try_from([1, 0, 2])?;
    /// assert_ne!(p.inverse(), q);
    /// assert_eq!(p.compose(&q)?.sigma, vec![0, 2, 1]);
    /// assert_ne!(p.compose(&q)?, q.compose(&p)?);
    ///
    /// let v = vec![-3, 4, 0];
    /// assert_eq!(p.compose(&q)?.apply(&v)?, q.apply(&p.apply(&v)?)?); 
    ///
    ///
    /// // If permutations are inverses of one another, their compositions are commutative
    /// let r = Permutation::try_from([2, 0, 1])?;
    /// assert_eq!(p.inverse(), r);
    /// assert_eq!(p.compose(&r)?, Permutation::identity(3));
    /// assert_eq!(r.compose(&p)?, Permutation::identity(3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn compose(&self, other: &Permutation) -> Result<Permutation, PermutationError> {
        Ok(Permutation {sigma: self.inverse().apply(&other.sigma)?})
    }

    pub fn iter_pairs(&self) -> std::iter::Enumerate<std::slice::Iter<u8>> {
        self.sigma.iter().enumerate()
    }
}

/// Implements indexing, letting Permutation behave as a container directly
impl Index<usize> for Permutation {
    type Output = u8;

    fn index(&self, i: usize) -> &Self::Output {
        &self.sigma[i]
    }
}

/// Generate a Permutation from an integer vector
///
/// ```
/// # use molassembler::permutation::Permutation;
/// # use std::convert::TryFrom;
/// let p = Permutation::try_from(vec![0, 1, 2]);
/// assert_eq!(p, Ok(Permutation::from_index(3, 0)));
/// let q = Permutation::try_from(vec![-1, 0, 1]);
/// assert!(q.is_err());
/// ```
impl<I: PrimInt> TryFrom<Vec<I>> for Permutation {
    type Error = PermutationError;

    fn try_from(integer_sigma: Vec<I>) -> Result<Permutation, Self::Error> {
        // Every number in 0..n must be present exactly once
        let n = integer_sigma.len();
        if n > u8::MAX as usize {
            return Err(PermutationError::NotRepresentable);
        }

        // Check to ensure values match expected
        let has_correct_set_elements = integer_sigma.iter()
            .sorted()
            .enumerate()
            .all(|(i, v)| v.to_usize().filter(|w| i == *w).is_some());

        if !has_correct_set_elements {
            return Err(PermutationError::InvalidSetElements);
        }

        let sigma = integer_sigma.iter().map(|v| v.to_u8().unwrap()).collect();
        Ok(Permutation {sigma})
    }
}

/// Generate a Permutation from an integer slice
/// ```
/// # use molassembler::permutation::Permutation;
/// # use std::convert::TryFrom;
/// let p = Permutation::try_from([0, 2, 1]);
/// assert_eq!(p, Ok(Permutation::from_index(3, 1)));
/// ```
impl<I: PrimInt, const N: usize> TryFrom<[I; N]> for Permutation {
    type Error = PermutationError;

    fn try_from(integer_slice: [I; N]) -> Result<Permutation, Self::Error> {
        Permutation::try_from(Vec::from(integer_slice))
    }
}

/// Iterator adaptor for permutations
pub struct PermutationIterator {
    permutation: Permutation,
    increment: bool
}

impl PermutationIterator {
    fn new(permutation: Permutation) -> PermutationIterator {
        PermutationIterator {permutation, increment:false}
    }
}

// TODO figure out how to return references to permutation with the appropriate lifetime (GATs)
impl Iterator for PermutationIterator {
    type Item = Permutation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.increment && !self.permutation.next_permutation() {
            return None;
        }

        self.increment = true;
        Some(self.permutation.clone())
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = Permutation::group_order(self.permutation.set_size()) - self.permutation.index();
        (remaining, Some(remaining))
    }
}

/// Yields permutations in increasing lexicographic order
///
/// ```
/// # use molassembler::permutation::{Permutation, permutations};
/// let mut iter = permutations(2);
/// assert_eq!(iter.next(), Some(Permutation::from_index(2, 0)));
/// assert_eq!(iter.next(), Some(Permutation::from_index(2, 1)));
/// assert_eq!(iter.next(), None);
/// ```
pub fn permutations(n: usize) -> PermutationIterator {
    PermutationIterator::new(Permutation::identity(n))
}

#[cfg(test)]
mod tests {
    use crate::permutation::*;

    #[test]
    fn small() {
        assert_eq!(Permutation::identity(0).sigma.len(), 0);
        assert_eq!(Permutation::from_index(0, 0).index(), 0);
        assert_eq!(Permutation::from_index(0, 1).index(), 0);
        assert_eq!(Permutation::identity(0).next_permutation(), false);
        assert_eq!(Permutation::identity(0).prev_permutation(), false);

        assert_eq!(Permutation::identity(1).sigma.len(), 1);
        assert_eq!(Permutation::from_index(1, 0).index(), 0);
        assert_eq!(Permutation::from_index(1, 1).index(), 0);
        assert_eq!(Permutation::identity(1).next_permutation(), false);
        assert_eq!(Permutation::identity(1).prev_permutation(), false);

        assert_eq!(Permutation::identity(2).sigma.len(), 2);
        assert_eq!(Permutation::from_index(2, 0).index(), 0);
        assert_eq!(Permutation::from_index(2, 1).index(), 1);
        assert_eq!(Permutation::from_index(2, 2).index(), 1);
        assert_eq!(Permutation::from_index(2, 3).index(), 1);
        assert_eq!(Permutation::from_index(2, 4).index(), 1);
    }

    fn random_discrete(n: usize) -> usize {
        let float = rand::random::<f32>();
        (float * n as f32) as usize
    }

    fn random_permutation(n: usize) -> Permutation {
        Permutation::from_index(n, random_discrete(Permutation::group_order(n)))
    }

    #[test]
    fn composition() -> Result<(), PermutationError> {
        let n = 6;
        let repeats = 100;

        let v: Vec<usize> = (0..n).map(|_| random_discrete(100)).collect();

        for _ in 0..repeats {
            let p = random_permutation(n);
            let q = random_permutation(n);

            let compose_apply = p.compose(&q)?.apply(&v)?;
            let apply_twice = q.apply(&p.apply(&v)?)?;
            assert_eq!(compose_apply, apply_twice);
        }

        Ok(())
    }

    #[test]
    fn application() -> Result<(), PermutationError> {
        let n = 6;
        let repeats = 100;

        let v: Vec<usize> = (0..n).map(|_| random_discrete(100)).collect();

        for _ in 0..repeats {
            let p = random_permutation(n);

            let w = p.apply(&v)?;
            let v_reconstructed = p.inverse().apply(&w)?;
            assert_eq!(v, v_reconstructed);
        }

        Ok(())
    }
}
