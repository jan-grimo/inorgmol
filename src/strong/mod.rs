use num_traits::int::PrimInt;
use num_traits::{FromPrimitive, ToPrimitive};

pub use index_derive::Index;

pub trait Index {
    type Type : PrimInt + ToPrimitive + FromPrimitive;

    fn get(&self) -> Self::Type;
}

// Collective trait
pub trait NewTypeIndex : 
    Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type>
    + PartialEq
{}

// Blanket impl for any Index types since NewTypeIndex doesn't define anything
impl<T> NewTypeIndex for T where T: Index 
    + Copy
    + From<<Self as Index>::Type> 
    + Into<<Self as Index>::Type> 
    + PartialEq
{}

pub mod matrix;
pub mod bijection;

#[cfg(test)]
mod tests {
    use crate::strong::Index;

    #[derive(Index, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Foo(u8);

    #[test]
    fn basics() {
        let a: Foo = 8.into();
        let _b: u8 = a.into();
        // let c: usize = a.into(); compilation failure
        let _d = Foo::from(3);
    }
}
