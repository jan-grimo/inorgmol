use num_traits::int::PrimInt;
use num_traits::FromPrimitive;

pub use index_derive::Index;

pub trait Index {
    type Type : PrimInt + FromPrimitive;

    fn get(&self) -> Self::Type;
}

// Collective trait
pub trait NewTypeIndex : 
    Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type>
{}

// Blanket impl to add From and Into if Index is defined
impl<T> NewTypeIndex for T where T: Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type> 
{}

#[cfg(test)]
mod tests {
    use crate::index::Index;
    use derive_more::{From, Into};

    #[derive(Index, From, Into, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Foo(u8);

    #[test]
    fn basics() {
        let a: Foo = 8.into();
        let _b: u8 = a.into();
        // let c: usize = a.into(); compilation failure
        let _d = Foo::from(3);
    }
}
