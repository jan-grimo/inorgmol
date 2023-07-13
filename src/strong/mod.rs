use num_traits::int::PrimInt;
use num_traits::{FromPrimitive, ToPrimitive};

pub use index_derive::IndexBase;

/// Base derivable trait for indexing with new types of integers
pub trait IndexBase {
    /// Underlying type
    type Type : PrimInt + ToPrimitive + FromPrimitive;

    /// Access the underlying type
    fn get(&self) -> Self::Type;
}

/// Collective trait used for indexing
pub trait Index : 
    IndexBase 
    + Copy
    + From<<Self as IndexBase>::Type> 
    + Into<<Self as IndexBase>::Type>
    + PartialEq
{}

// Blanket impl for any Index types since NewTypeIndex doesn't define anything
impl<T> Index for T where T: IndexBase 
    + Copy
    + From<<Self as IndexBase>::Type> 
    + Into<<Self as IndexBase>::Type> 
    + PartialEq
{}

/// Matrices indexed by new types
pub mod matrix;
/// Mappings between new type indices
pub mod bijection;

#[cfg(test)]
mod tests {
    use crate::strong::IndexBase;

    #[derive(IndexBase, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Foo(u8);

    #[test]
    fn basics() {
        let a: Foo = 8.into();
        let _b: u8 = a.into();
        // let c: usize = a.into(); compilation failure
        let _d = Foo::from(3);
    }
}
