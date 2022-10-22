use std::marker::PhantomData;
use num_traits::int::PrimInt;
use num_traits::ToPrimitive;
use num_traits::FromPrimitive;

use crate::permutation::{Permutation, PermutationError};

pub trait Index {
    type Type : PrimInt + FromPrimitive;

    fn get(&self) -> Self::Type;
}

pub trait NewTypeIndex : 
    Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type> 
{}

impl<T> NewTypeIndex for T where T: Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type> 
{}

pub struct Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
    pub permutation: Permutation,
    key_type: PhantomData<Key>,
    value_type: PhantomData<Value>
}

impl<Key, Value> Bijection<Key, Value> where Key: NewTypeIndex, Value: NewTypeIndex {
    pub fn new(p: Permutation) -> Bijection<Key, Value> {
        Bijection {permutation: p, key_type: PhantomData, value_type: PhantomData}
    }

    pub fn inverse(&self) -> Bijection<Value, Key> {
        Bijection::new(self.permutation.inverse())
    }

    pub fn get(&self, key: &Key) -> Option<Value> {
        let index = key.get().to_usize()?;
        let value = self.permutation.sigma.get(index)?;
        Some(Value::from(<Value::Type as FromPrimitive>::from_u8(*value)?))
    }

    pub fn index(&self, value: &Value) -> Option<Key> {
        let v = value.get().to_u8()?;
        let v_position = self.permutation.sigma.iter().position(|x| *x == v)?;
        Some(Key::from(<Key::Type as FromPrimitive>::from_usize(v_position)?))
    }

    pub fn compose<OtherValue>(&self, other: &Bijection<Value, OtherValue>) -> Result<Bijection<Key, OtherValue>, PermutationError> where OtherValue: NewTypeIndex {
        let p = self.permutation.compose(&other.permutation)?;
        Ok(Bijection::new(p))
    }
}

#[cfg(test)]
mod tests {
    use derive_more::{From, Into};
    use crate::permutation::Permutation;
    use crate::bijection::{Index, Bijection};

    #[derive(From, Into, Debug, Copy, Clone, PartialEq)]
    struct Foo(u8);

    #[derive(From, Into, Debug, Copy, Clone, PartialEq)]
    struct Bar(u8);

    #[derive(From, Into, Debug, Copy, Clone, PartialEq)]
    struct Baz(u8);

    impl Index for Foo {
        type Type = u8;

        fn get(&self) -> u8 { self.0 }
    }

    impl Index for Bar {
        type Type = u8;

        fn get(&self) -> u8 { self.0 }
    }

    impl Index for Baz {
        type Type = u8;

        fn get(&self) -> u8 { self.0 }
    }

    #[test]
    fn basics() {
        let f1 = Bijection::<Foo, Bar>::new(Permutation::from_index(4, 3));
        let f1_at_two = Bar::from(f1.permutation.sigma[2]);
        assert_eq!(f1.get(&Foo(2)), Some(f1_at_two));
        assert_eq!(f1.index(&f1_at_two), Some(Foo::from(2)));

        let f2 = Bijection::<Bar, Baz>::new(Permutation::from_index(4, 3));
        let f3 = f1.compose(&f2).unwrap();
        assert_eq!(f3.permutation, f1.permutation.compose(&f2.permutation).unwrap());

        // let f4 = f1.compose(&f3); expected compile error (mismatched types)

        let p1 = Bijection::<Bar, Bar>::new(Permutation::identity(4));
        let f4 = f1.compose(&p1).unwrap();
        assert_eq!(f1.permutation, f4.permutation);
    }
}
