use std::marker::PhantomData;
use num_traits::ToPrimitive;
use num_traits::FromPrimitive;
use delegate::delegate;

use crate::permutation::{Permutation, PermutationError};
use crate::strong::NewTypeIndex;

#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Debug, Hash)]
pub struct Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
    pub permutation: Permutation,
    key_type: PhantomData<Key>,
    value_type: PhantomData<Value>
}

impl<Key, Value> Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
    /// Initialize by wrapping a Permutation
    pub fn new(p: Permutation) -> Bijection<Key, Value> {
        Bijection {permutation: p, key_type: PhantomData, value_type: PhantomData}
    }

    pub fn new_random(n: usize) -> Bijection<Key, Value> {
        Bijection::new(Permutation::new_random(n))
    }

    pub fn try_from_index(n: usize, i: usize) -> Option<Bijection<Key, Value>> {
        Permutation::try_from_index(n, i).map(Bijection::new)
    }

    pub fn identity(n: usize) -> Bijection<Key, Value> {
        Bijection::new(Permutation::identity(n))
    }

    /// Invert the bijection
    pub fn inverse(&self) -> Bijection<Value, Key> {
        Bijection::new(self.permutation.inverse())
    }

    /// Apply the map to a key and find its corresponding value
    pub fn get(&self, key: &Key) -> Option<Value> {
        let index = key.get().to_usize()?;
        let value = self.permutation[index];
        Some(Value::from(<Value::Type as FromPrimitive>::from_usize(value)?))
    }

    /// Find the key to a corresponding value
    pub fn inverse_of(&self, value: &Value) -> Option<Key> {
        let inverse = self.permutation.inverse_of(value.get().to_usize()?)?;
        let key = Key::from(<Key::Type as FromPrimitive>::from_usize(inverse)?);
        Some(key)
    }

    /// Compose the bijection with another
    pub fn compose<OtherValue>(&self, other: &Bijection<Value, OtherValue>) -> Result<Bijection<Key, OtherValue>, PermutationError> where OtherValue: NewTypeIndex {
        let p = self.permutation.compose(&other.permutation)?;
        Ok(Bijection::new(p))
    }

    delegate! {
        to self.permutation {
            pub fn index(&self) -> usize;
            pub fn next_permutation(&mut self) -> bool;
            pub fn prev_permutation(&mut self) -> bool;
            pub fn set_size(&self) -> usize;
        }
    }

    pub fn group_order(n: usize) -> usize {
        Permutation::group_order(n)
    }

    pub fn is_fixed_point(&self, key: Key) -> bool {
        self.permutation.is_fixed_point(key.get().to_usize().unwrap())
    }
}

impl<Key, Value> std::fmt::Display for Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.permutation)
    }
}

pub struct BijectionIterator<T, U> where T: NewTypeIndex, U: NewTypeIndex {
    bijection: Bijection<T, U>,
    increment: bool
}

impl<T, U> BijectionIterator<T, U> where T: NewTypeIndex, U: NewTypeIndex {
    fn new(bijection: Bijection<T, U>) -> BijectionIterator<T, U> {
        BijectionIterator {bijection, increment: false}
    }
}

impl<T, U> Iterator for BijectionIterator<T, U> where T: NewTypeIndex, U: NewTypeIndex {
    type Item = Bijection<T, U>;

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

pub fn bijections<T, U>(n: usize) -> BijectionIterator<T, U> where T: NewTypeIndex, U: NewTypeIndex {
    BijectionIterator::<T, U>::new(Bijection::new(Permutation::identity(n)))
}

#[cfg(test)]
mod tests {
    use crate::permutation::Permutation;
    use crate::strong::bijection::Bijection;
    use crate::strong::Index;

    #[derive(Index, Debug, Copy, Clone, PartialEq)]
    struct Foo(usize);

    #[derive(Index, Debug, Copy, Clone, PartialEq)]
    struct Bar(usize);

    #[derive(Index, Debug, Copy, Clone, PartialEq)]
    struct Baz(usize);

    #[test]
    fn basics() {
        let f1 = Bijection::<Foo, Bar>::new(Permutation::try_from_index(4, 3).expect("Valid index"));
        let f1_at_two = Bar::from(f1.permutation[2]);
        assert_eq!(f1.get(&Foo(2)), Some(f1_at_two));
        assert_eq!(f1.inverse_of(&f1_at_two), Some(Foo::from(2)));

        let f2 = Bijection::<Bar, Baz>::new(Permutation::try_from_index(4, 3).expect("Valid index"));
        let f3 = f1.compose(&f2).unwrap();
        assert_eq!(f3.permutation, f1.permutation.compose(&f2.permutation).unwrap());

        // let f4 = f1.compose(&f3); expected compile error (mismatched types)

        let p1 = Bijection::<Bar, Bar>::new(Permutation::identity(4));
        let f4 = f1.compose(&p1).unwrap();
        assert_eq!(f1.permutation, f4.permutation);
    }
}
