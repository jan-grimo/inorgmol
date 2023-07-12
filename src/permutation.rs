use std::cmp::Ordering;
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

/// Apply a permutation to a slice of values
pub fn slice_apply<T: PrimInt, U: Copy>(permutation: &[T], values: &[U]) -> Option<Vec<U>> {
    let n = permutation.len();
    if n != values.len() {
        return None;
    } 

    let mut permuted = values.to_vec();
    for (i, &value) in values.iter().enumerate() {
        let target = permutation[i].to_usize().unwrap();
        permuted[target] = value;
    }
    Some(permuted)
}

fn random_discrete(n: usize) -> usize {
    let float = rand::random::<f32>();
    (float * n as f32) as usize
}

/// Struct representing an order permutation
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct Permutation {
    /// One-line representation
    sigma: Vec<usize>
}

impl std::fmt::Display for Permutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{}}}", self.sigma.iter().format(", "))
    }
}

/// Errors arising in use of permutations
#[derive(Error, Debug, PartialEq, Eq)]
pub enum PermutationError {
    /// Permutation and argument have mismatching lengths
    #[error("Mismatched lengths between permutation and argument")]
    LengthMismatch,
    /// Elements of one-line representation are invalid
    #[error("Invalid elements of one-line representation set")]
    InvalidSetElements
}

impl Permutation {
    /// Initialize an identity permutation of specific size
    ///
    /// The identity permutation maps each number onto itself. It has index
    /// zero within the lexicographical order of permutations.
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation::identity(3).sigma, vec![0, 1, 2])
    /// ```
    pub fn identity(n: usize) -> Permutation {
        Permutation {sigma: (0..n).collect()}
    }

    /// Constructs a new permutation without ensuring the result is valid
    pub fn new_unchecked(v: Vec<usize>) -> Permutation {
        Permutation {sigma: v}
    }

    /// Initialize the `i`-th permutation by lexicographic order of size `n`
    ///
    /// Initializes a particular permutation identified by its position
    /// (zero-indexed) in the lexicographic order `i` of all permutations of
    /// size `n`.
    ///
    /// If `i` is larger than the final position of all permutations of that
    /// size, i.e. $ i >= n! $, `None` is returned.
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation::try_from_index(2, 0), Some(Permutation {sigma: vec![0, 1]}));
    /// assert_eq!(Permutation::try_from_index(2, 1), Some(Permutation {sigma: vec![1, 0]}));
    /// assert_eq!(Permutation::try_from_index(2, 2), None);
    /// ```
    pub fn try_from_index(n: usize, mut i: usize) -> Option<Permutation> {
        let mut factorials = Vec::<usize>::with_capacity(n);
        factorials.push(1);
        for k in 1..n {
            factorials.push(factorials.last().unwrap() * k);
        }

        if i >= *factorials.last().unwrap() * n {
            return None;
        }

        let mut sigma = Vec::with_capacity(n);

        for k in 0..n {
            let fac = factorials[n - 1 - k];
            sigma.push(i / fac);
            i %= fac;
        }

        for k in (1..n).rev() {
            for j in (0..k).rev() {
                if sigma[j] <= sigma[k] {
                    sigma[k] += 1;
                }
            }
        }

        Some(Permutation {sigma})
    }

    /// Find a permutation ordering a slice's elements
    pub fn ordering_by_key<T, F, U>(slice: &[T], key: F) -> Permutation 
        where F: Fn(&T) -> U, U: Ord
    {
        let mut p = Permutation::identity(slice.len());
        p.sigma.sort_by_key(|&i| key(&slice[i]));
        p.inverse()
    }

    /// Find a permutation ordering a slice's elements
    pub fn ordering_by<T, F>(slice: &[T], compare: F) -> Permutation 
        where F: Fn(&T, &T) -> Ordering
    {
        let mut p = Permutation::identity(slice.len());
        p.sigma.sort_by(|&i, &j| compare(&slice[i], &slice[j]));
        p.inverse()
    }

    /// Find a permutation ordering a slice's elements
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let container = vec![-3, 2, 0];
    /// let p = Permutation::ordering(container.as_slice());
    /// assert_eq!(p, Permutation::try_from_index(3, 1).unwrap());
    /// assert_eq!(p.apply(container), Ok(vec![-3, 0, 2]));
    /// ```
    pub fn ordering<T: Ord>(slice: &[T]) -> Permutation {
        Self::ordering_by(slice, |a, b| a.cmp(b))
    }

    /// Determine the index of a permutation in its lexicographic order
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// assert_eq!(Permutation {sigma: vec![0, 1, 2]}.index(), 0);
    /// assert_eq!(Permutation {sigma: vec![0, 2, 1]}.index(), 1);
    ///
    /// let num_permutations_size_five = Permutation::group_order(5);
    /// for i in 0..num_permutations_size_five {
    ///     if let Some(p) = Permutation::try_from_index(5, i) {
    ///         assert_eq!(p.index(), i);
    ///     }
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
    /// let mut permutation = Permutation::try_from_index(6, 0).expect("Valid index");
    /// for i in 1..15 {
    ///     assert_eq!(permutation.next_permutation(), true);
    ///     assert_eq!(permutation, Permutation::try_from_index(6, i).expect("Valid index"));
    /// }
    /// ```
    pub fn next_permutation(&mut self) -> bool {
        slice_next(self.sigma.as_mut_slice())
    }

    /// Transform into the previous permutation within its set's partial order
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let mut permutation = Permutation::try_from_index(6, 15).expect("Valid index");
    /// for i in 14..1 {
    ///     assert_eq!(permutation.prev_permutation(), true);
    ///     assert_eq!(permutation, Permutation::try_from_index(6, i).expect("Valid index"));
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
    /// let permutation = Permutation::try_from_index(3, 1).expect("Valid index");
    /// assert_eq!(permutation.inverse().compose(&permutation)?, Permutation::identity(3));
    /// assert_eq!(permutation.compose(&permutation.inverse())?, Permutation::identity(3));
    /// # Ok(())
    /// # }
    /// ```
    pub fn inverse(&self) -> Permutation {
        let n = self.sigma.len();
        let mut inverse = Permutation::identity(n);
        for i in 0..n {
            inverse.sigma[self.sigma[i]] = i;
        }
        inverse
    }

    /// Finds `j` so that `self[i] == j`
    pub fn inverse_of(&self, index: usize) -> Option<usize> {
        self.sigma.iter().position(|x| *x == index)
    }

    /// Apply the permutation to a vector
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// let p = Permutation::try_from([1, 2, 0])?;
    /// let v = vec!["I", "am", "Yoda"];
    /// assert_eq!(p.apply(v), Ok(vec!["Yoda", "I", "am"]));
    /// # Ok(())
    /// # }
    /// ```
    pub fn apply<T>(&self, values: Vec<T>) -> Result<Vec<T>, PermutationError> {
        let n = self.set_size();
        if n != values.len() {
            return Err(PermutationError::LengthMismatch);
        }

        let mut permuted_maybe = Vec::with_capacity(n);
        permuted_maybe.resize_with(n, || std::mem::MaybeUninit::<T>::uninit());

        for (i, value) in values.into_iter().enumerate() {
            permuted_maybe[self[i]].write(value);
        }

        // SAFETY: Permutation guarantees contiguous numbers on construction, each
        // item in permuted_maybe will be written to
        Ok(unsafe {std::mem::transmute(permuted_maybe) })
    }

    /// Apply the permutation
    pub fn apply_ref<T: Copy>(&self, values: &Vec<T>) -> Result<Vec<T>, PermutationError> {
        slice_apply(self.sigma.as_slice(), values.as_slice()).ok_or(PermutationError::LengthMismatch)
    }

    /// Apply the permutation to a slice of values and collect the result
    pub fn apply_collect<B, T: Copy>(&self, slice: &[T]) -> Result<B, PermutationError> 
        where B: FromIterator<T>
    {
        slice_apply(self.sigma.as_slice(), slice)
            .map(|v| B::from_iter(v.into_iter()))
            .ok_or(PermutationError::LengthMismatch)
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
    /// assert_eq!(p.compose(&q)?.apply_ref(&v)?, q.apply(p.apply_ref(&v)?)?); 
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
        Ok(Permutation {sigma: self.inverse().apply_ref(&other.sigma)?})
    }

    /// Iterate through the one-line representation in `(i, sigma[i])` pairs
    pub fn iter_pairs(&self) -> std::iter::Enumerate<std::slice::Iter<usize>> {
        self.sigma.iter().enumerate()
    }

    /// Generate a random permutation (identity inclusive)
    pub fn new_random(n: usize) -> Permutation {
        Permutation::try_from_index(n, random_discrete(Permutation::group_order(n))).expect("Proposed valid permutation")
    }

    /// Fixed points are values that the permutation does not permute
    pub fn is_fixed_point(&self, i: usize) -> bool {
        self.sigma[i] == i
    }

    /// Check whether a permutation is the identity permutation
    pub fn is_identity(&self) -> bool {
        self.sigma.iter().enumerate().all(|(i, &v)| i == v)
    }

    /// Derangements permute all elements, i.e. there are no fixed points
    pub fn is_derangement(&self) -> bool {
        self.iter_pairs().all(|(i, &v)| i != v)
    }

    /// Destructure a permutation into cycles, not including fixed points
    pub fn cycles(&self) -> Vec<Vec<usize>> {
        let n = self.set_size();
        let mut element_found = vec![false; n];
        let mut cycles = Vec::new();

        for i in 0..n {
            if element_found[i] {
                continue;
            }

            let mut cycle = vec![i];
            element_found[i] = true;
            let mut item = self.sigma[i];
            while !element_found[item] {
                cycle.push(item);
                element_found[item] = true;
                item = self.sigma[item];
            }

            if cycle.len() > 1 {
                cycles.push(cycle);
            }
        }

        cycles
    }

    /// Increase set size of permutation by pushing onto one-line representation
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// let mut p = Permutation::try_from([1, 2, 0])?;
    /// assert_eq!(p.set_size(), 3);
    /// p.push();
    /// assert_eq!(p.set_size(), 4);
    /// assert_eq!(p, Permutation::try_from([1, 2, 0, 3])?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn push(&mut self) {
        self.sigma.push(self.sigma.len());
    }

    /// If the last element maps to itself, pops and returns it
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// let mut p = Permutation::try_from([1, 2, 0, 3])?;
    /// assert_eq!(p.pop_if_fixed_point(), Some(3));
    /// let mut q = Permutation::try_from([1, 2, 3, 0])?;
    /// assert_eq!(q.pop_if_fixed_point(), None);
    /// # Ok(())
    /// # }
    /// ```
    pub fn pop_if_fixed_point(&mut self) -> Option<usize> {
        if self.sigma.last().filter(|&&v| v == self.sigma.len() - 1).is_some() {
            self.sigma.pop()
        } else {
            None
        }
    }

    /// Advance a slice of the one-line representation to its next permutation
    ///
    /// The slice of the one-line representation to advance can be indicated by any range
    /// expression.
    ///
    /// ```
    /// # use molassembler::permutation::{Permutation, PermutationError};
    /// # use std::convert::TryFrom;
    /// # fn main() -> Result<(), PermutationError> {
    /// let mut p = Permutation::identity(4);
    /// assert!(p.slice_next(1..2));
    /// assert_eq!(p, Permutation::try_from([0, 2, 1, 3])?);
    /// # Ok(())
    /// # }
    /// ```
    pub fn slice_next<R: std::slice::SliceIndex<[usize], Output = [usize]>>(&mut self, range: R) -> bool {
        slice_next(&mut self.sigma[range])
    }

    /// Decrement a slice of the one-line representation to its previous permutation
    ///
    /// See [`Permutation::slice_next`] and [`slice_prev`]
    pub fn slice_prev<R: std::slice::SliceIndex<[usize], Output = [usize]>>(&mut self, range: R) -> bool {
        slice_prev(&mut self.sigma[range])
    }
}

/// Implements indexing, letting Permutation behave as a container directly
impl Index<usize> for Permutation {
    type Output = usize;

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
/// assert_eq!(p, Ok(Permutation::try_from_index(3, 0).expect("Valid index")));
/// let q = Permutation::try_from(vec![-1, 0, 1]);
/// assert!(q.is_err());
/// ```
impl<I: PrimInt> TryFrom<Vec<I>> for Permutation {
    type Error = PermutationError;

    fn try_from(integer_sigma: Vec<I>) -> Result<Permutation, Self::Error> {
        // Every number in 0..n must be present exactly once
        let has_correct_set_elements = integer_sigma.iter()
            .sorted()
            .enumerate()
            .all(|(i, v)| v.to_usize().is_some_and(|w| i == w));

        if !has_correct_set_elements {
            return Err(PermutationError::InvalidSetElements);
        }

        let sigma = integer_sigma.iter().map(|v| v.to_usize().unwrap()).collect();
        Ok(Permutation {sigma})
    }
}

/// Generate a Permutation from an integer array
///
/// ```
/// # use molassembler::permutation::Permutation;
/// # use std::convert::TryFrom;
/// let p = Permutation::try_from([0, 2, 1]);
/// assert_eq!(p, Ok(Permutation::try_from_index(3, 1).expect("Valid index")));
/// ```
impl<I: PrimInt, const N: usize> TryFrom<[I; N]> for Permutation {
    type Error = PermutationError;

    fn try_from(integer_slice: [I; N]) -> Result<Permutation, Self::Error> {
        Permutation::try_from(Vec::from(integer_slice))
    }
}

/// Iterator adaptor for iterating through all permutations of a set size
///
/// See [`permutations`]
#[derive(Clone)]
pub struct PermutationIterator {
    permutation: Permutation,
    increment: bool
}

impl PermutationIterator {
    fn new(permutation: Permutation) -> PermutationIterator {
        PermutationIterator {permutation, increment:false}
    }
}

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
/// assert_eq!(iter.next().map(|p| p.index()), Some(0));
/// assert_eq!(iter.next().map(|p| p.index()), Some(1));
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
        assert_eq!(Permutation::try_from_index(0, 0), None);
        assert!(!Permutation::identity(0).next_permutation());
        assert!(!Permutation::identity(0).prev_permutation());

        assert_eq!(Permutation::identity(1).sigma.len(), 1);
        assert_eq!(Permutation::try_from_index(1, 0).expect("Valid index").index(), 0);
        assert_eq!(Permutation::try_from_index(1, 1), None);
        assert!(!Permutation::identity(1).next_permutation());
        assert!(!Permutation::identity(1).prev_permutation());

        assert_eq!(Permutation::identity(2).sigma.len(), 2);
        assert_eq!(Permutation::try_from_index(2, 0).expect("Valid index").index(), 0);
        assert_eq!(Permutation::try_from_index(2, 1).expect("Valid index").index(), 1);
        assert_eq!(Permutation::try_from_index(2, 2), None);
    }

    #[test]
    fn composition() -> Result<(), PermutationError> {
        let n = 6;
        let repeats = 100;

        let v: Vec<usize> = (0..n).map(|_| random_discrete(100)).collect();

        for _ in 0..repeats {
            let p = Permutation::new_random(n);
            let q = Permutation::new_random(n);

            let compose_apply = p.compose(&q)?.apply_ref(&v)?;
            let apply_twice = q.apply_ref(&p.apply_ref(&v)?)?;
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
            let p = Permutation::new_random(n);

            let w = p.apply_ref(&v)?;
            let v_reconstructed = p.inverse().apply_ref(&w)?;
            assert_eq!(v, v_reconstructed);
        }

        Ok(())
    }

    #[test]
    fn ordering_permutation() -> Result<(), PermutationError> {
        let indices: Vec<usize> = (0..10).collect();
        let disordering = Permutation::new_random(indices.len());
        let disordered = disordering.apply_ref(&indices)?;

        let ordering = Permutation::ordering(&disordered);
        let ordered = ordering.apply_ref(&disordered)?;

        assert_eq!(indices, ordered);
        assert_eq!(disordering.inverse(), ordering);

        Ok(())
    }

    #[test]
    fn range_equal() {
        itertools::assert_equal(
            permutations(4),
            (0..4).permutations(4).map(|p| Permutation {sigma: p})
        );
    }

    #[test]
    fn cycles_correct() {
        let perm = Permutation {sigma: vec![1, 4, 3, 2, 0]};
        let expected_cycles = vec![vec![0, 1, 4], vec![2, 3]];
        assert!(perm.cycles() == expected_cycles);
    }
}
