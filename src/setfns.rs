use itertools::Itertools;
use thiserror::Error;
use std::hash::Hash;

/// Errors involving discrete fns
#[derive(Error, Debug, PartialEq, Eq)]
pub enum DiscreteFnError {
    /// Passed a non-bijective set function
    #[error("Not a bijective set function")]
    NotBijective,
}

/// A surjection implementation for few elements
#[derive(Clone, Debug)]
pub struct TinySurjection<T, U> {
    domain: Vec<T>,
    codomain: Vec<U>
}

impl<T, U> FromIterator<(T, U)> for TinySurjection<T, U> 
    where T: Clone + Ord + Hash
{
    fn from_iter<A: IntoIterator<Item = (T, U)>>(collection: A) -> Self {
        let (domain, codomain) = collection.into_iter()
            .unique_by(|(a, _)| (*a).clone())
            .sorted_by(|a, b| Ord::cmp(&a.0, &b.0))
            .unzip();

        Self {domain, codomain}
    }
}

/// A bijection implementation for few elements
#[derive(Clone, Debug)]
pub struct TinyBijection<T, U> {
    domain: Vec<T>,
    codomain: Vec<U>
}

impl<T, U> TinyBijection<T, U> 
where T: Clone + Ord + Hash,
    U: Eq + Hash
{
    /// Try to convert a collection of pairs in to a bijection
    pub fn try_from_iter<A: IntoIterator<Item=(T, U)>>(collection: A) -> Result<Self, DiscreteFnError> {
        let (domain, codomain): (Vec<_>, Vec<_>) = collection.into_iter()
            .unique_by(|(a, _)| (*a).clone())
            .sorted_by_key(|(a, _)| (*a).clone())
            .unzip();
        if codomain.iter().unique().count() < domain.len() {
            Err(DiscreteFnError::NotBijective)
        } else {
            Ok(Self {domain, codomain})
        }
    }
}

/// Surjective discrete function
pub trait Surjection {
    /// Source / key / input type
    type Domain: Clone + Ord;
    /// Target / value / output type
    type Codomain: Clone + Ord + Hash;

    /// Type of an iterator yielding pairs of references to the mapped values
    type PairIterator<'a>: Iterator<Item=(&'a Self::Domain, &'a Self::Codomain)> where Self: 'a;

    /// Unordered iterator of all mapped pairs
    fn pairs(&self) -> Self::PairIterator<'_>;

    /// Size of the input domain
    fn domain_size(&self) -> usize {
        self.pairs().count()
    }

    /// Size of the output domain
    fn codomain_size(&self) -> usize {
        self.pairs().map(|(_, v)| v).unique().count()
    }

    /// Fetch a codomain value for a domain value, if present
    fn get(&self, key: &Self::Domain) -> Option<&Self::Codomain> {
        self.pairs().find(|(a, _)| *a == key).map(|(_, value)| value)
    }

    /// Compose with another surjection. 
    ///
    /// Values in `self` not mapped by `other` are missing in the composed surjection
    fn compose<B, Other>(&self, other: &Other) -> B
    where Other: Surjection<Domain=Self::Codomain>,
        B: Surjection<Domain=Self::Domain, Codomain=<Other as Surjection>::Codomain> + FromIterator<(Self::Domain, <Other as Surjection>::Codomain)>
    {
        let iter = self.pairs()
            .filter_map(|(a, b)| other.get(b).map(|v| ((*a).clone(), (*v).clone())));

        B::from_iter(iter)
    }
}

impl<T, U> Surjection for TinySurjection<T, U> 
    where T: Clone + Ord, U: Clone + Ord + Hash
{
    type Domain = T;
    type Codomain = U;

    type PairIterator<'a> = std::iter::Zip<std::slice::Iter<'a, T>, std::slice::Iter<'a, U>> where T: 'a, U: 'a;

    fn pairs(&self) -> Self::PairIterator<'_> {
        self.domain.iter().zip(self.codomain.iter())
    }
}

impl<T, U> AsRef<[U]> for TinySurjection<T, U>
    where U: Clone + Ord + Hash 
{
    fn as_ref(&self) -> &[U] {
        self.codomain.as_slice()
    }
}

impl<T, U> AsMut<[U]> for TinySurjection<T, U>
    where U: Clone + Ord + Hash
{
    fn as_mut(&mut self) -> &mut [U] {
        self.codomain.as_mut_slice()
    }
}

impl<T, U> Surjection for TinyBijection<T, U> 
    where T: Clone + Ord, U: Clone + Ord + Hash
{
    type Domain = T;
    type Codomain = U;

    type PairIterator<'a> = std::iter::Zip<std::slice::Iter<'a, T>, std::slice::Iter<'a, U>> where T: 'a, U: 'a;

    fn pairs(&self) -> Self::PairIterator<'_> {
        self.domain.iter().zip(self.codomain.iter())
    }
}

impl<T, U> AsRef<[U]> for TinyBijection<T, U>
    where U: Clone + Ord + Hash 
{
    fn as_ref(&self) -> &[U] {
        self.codomain.as_slice()
    }
}

impl<T, U> AsMut<[U]> for TinyBijection<T, U>
    where U: Clone + Ord + Hash
{
    fn as_mut(&mut self) -> &mut [U] {
        self.codomain.as_mut_slice()
    }
}

/// Bijective discrete function
pub trait Bijection: Surjection where Self: Clone {
    /// Type of the inverted bijection
    type Inverse: Bijection;

    /// Fetch a domain value for a codomain value, if present
    fn inverse_of(&self, value: &<Self as Surjection>::Codomain) -> Option<&<Self as Surjection>::Domain> {
        self.pairs().find(|(_, a)| *a == value).map(|(key, _)| key)
    }

    /// Invert the bijection, consuming it
    fn invert(self) -> Self::Inverse;

    /// Generate the inverse of the bijection
    fn inverse(&self) -> Self::Inverse {
        self.clone().invert()
    }
}

impl<T, U> Bijection for TinyBijection<T, U> 
    where T: Clone + Ord + Hash, U: Clone + Ord + Hash
{
    type Inverse = TinyBijection<U, T>;

    fn invert(self) -> Self::Inverse {
        Self::Inverse {domain: self.codomain, codomain: self.domain}
    }
}

#[cfg(test)]
mod tests {
    use crate::setfns::*;
    use crate::permutation::*;
    use crate::strong::{Index, IndexBase};

    #[derive(IndexBase, Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
    struct Foo(u8);

    #[derive(IndexBase, Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
    struct Bar(u8);

    #[test]
    fn index_compatible() {
        let a = Foo::range(10).skip(4).take(5); // [4, 5, 6, 7, 8]
        let b = Bar::range(10).skip(2).take(5); // [2, 3, 4, 5, 6]

        let mut surjection = TinySurjection::from_iter(a.zip(b));
        assert_eq!(surjection.permutation_index(), 0);

        assert_eq!(surjection.get(&Foo::from(7)), Some(&Bar::from(5)));

        assert!(surjection.next_permutation());
        assert_eq!(surjection.permutation_index(), 1);
    }
}
