use std::ops::Index;
use itertools::Itertools;

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

#[derive(PartialEq, Eq, PartialOrd, Clone, Debug)]
pub struct Permutation {
    // One-line representation
    pub sigma: Vec<u8>
}

impl std::fmt::Display for Permutation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{{{}}}", self.sigma.iter().format(", "))
    }
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
        let mut sigma: Vec<u8> = Vec::with_capacity(n);
        for i in 0..n as u8 {
            sigma.push(i)
        }
        Permutation {sigma}
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

    /// Transform into the next permutation within the partial order of its set
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let mut permutation = Permutation::from_index(6, 0);
    /// for i in 1..15 {
    ///     assert_eq!(permutation.next(), true);
    ///     assert_eq!(permutation, Permutation::from_index(6, i));
    /// }
    /// ```
    pub fn next(&mut self) -> bool {
        slice_next(self.sigma.as_mut_slice())
    }

    /// Transform into the previous permutation within its set's partial order
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// let mut permutation = Permutation::from_index(6, 15);
    /// for i in 14..1 {
    ///     assert_eq!(permutation.prev(), true);
    ///     assert_eq!(permutation, Permutation::from_index(6, i));
    /// }
    /// ```
    pub fn prev(&mut self) -> bool {
        slice_prev(self.sigma.as_mut_slice())
    }

    /// Invert the permutation
    ///
    /// ```
    /// # use molassembler::permutation::Permutation;
    /// # use std::result::Result;
    /// # fn main() -> Result<(), &'static str> {
    /// let permutation = Permutation::from_index(3, 1);
    /// assert_eq!(permutation.inverse().compose(&permutation)?, Permutation::identity(3));
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
    pub fn apply<T: Copy>(&self, other: &Vec<T>) -> Result<Vec<T>, &'static str> {
        let other_len = other.len();
        if other_len > self.sigma.len() {
            return Err("Permutation too short for passed container.");
        } 

        let mut permuted = other.clone();
        for i in 0..other_len {
            permuted[self.sigma[i] as usize] = other[i];
        }
        Ok(permuted)
    }

    /// Compose two permutations into a new permutation
    pub fn compose(&self, other: &Permutation) -> Result<Permutation, &'static str> {
        if self.sigma.len() != other.sigma.len() {
            return Err("Mismatched permutation sizes");
        }

        return Ok(Permutation {sigma: self.inverse().apply(&other.sigma)?});
    }

    pub fn iter(&mut self) -> PermutationIterator {
        PermutationIterator {permutation: self, increment: false}
    }
}

/// Implements indexing, letting Permutation behave as a container directly
impl Index<usize> for Permutation {
    type Output = u8;

    fn index(&self, i: usize) -> &Self::Output {
        &self.sigma[i]
    }
}

/// Iterator adaptor for permutations
///
/// ```
/// # use molassembler::permutation::Permutation;
/// let mut permutation = Permutation::from_index(2, 0);
/// let mut iter = permutation.iter();
/// assert_eq!(iter.next(), Some(Permutation::from_index(2, 0)));
/// assert_eq!(iter.next(), Some(Permutation::from_index(2, 1)));
/// assert_eq!(iter.next(), None);
/// ```
pub struct PermutationIterator<'a> {
    permutation: &'a mut Permutation,
    increment: bool
}

// TODO figure out how to return references to permutation with the appropriate lifetime
impl Iterator for PermutationIterator<'_> {
    type Item = Permutation;

    fn next(&mut self) -> Option<Self::Item> {
        if self.increment {
            if !self.permutation.next() {
                return None;
            }
        }

        self.increment = true;
        Some(self.permutation.clone())
    }
}

#[cfg(test)]
mod tests {
    use crate::permutation::*;

    #[test]
    fn small_permutations() {
        assert_eq!(Permutation::identity(0).sigma.len(), 0);
        assert_eq!(Permutation::from_index(0, 0).index(), 0);
        assert_eq!(Permutation::identity(0).next(), false);
        assert_eq!(Permutation::identity(0).prev(), false);

        assert_eq!(Permutation::identity(1).sigma.len(), 1);
        assert_eq!(Permutation::from_index(1, 0).index(), 0);
        assert_eq!(Permutation::identity(1).next(), false);
        assert_eq!(Permutation::identity(1).prev(), false);

        assert_eq!(Permutation::identity(2).sigma.len(), 2);
        assert_eq!(Permutation::from_index(2, 0).index(), 0);
        assert_eq!(Permutation::from_index(2, 1).index(), 1);
    }
}
