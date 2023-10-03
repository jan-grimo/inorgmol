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
pub trait Index: 
    IndexBase 
    + Copy
    + From<<Self as IndexBase>::Type> 
    + Into<<Self as IndexBase>::Type>
    + PartialEq
{
    /// Return a range with self as the upper bound
    fn range(bound: <Self as IndexBase>::Type) -> Range<Self>;
}

// Blanket impl Index for IndexBase types
impl<T> Index for T where T: IndexBase 
    + Copy
    + From<<Self as IndexBase>::Type> 
    + Into<<Self as IndexBase>::Type> 
    + PartialEq
{
    fn range(bound: <Self as IndexBase>::Type) -> Range<Self> {
        let start = <<T as IndexBase>::Type as num_traits::identities::Zero>::zero();
        Range {start, end: bound}
    }
}

/// Range generating Index Items
pub struct Range<I: Index> {
    start: <I as IndexBase>::Type,
    end: <I as IndexBase>::Type
}

impl<I: Index> Iterator for Range<I> {
    type Item = I;

    fn next(&mut self) -> Option<Self::Item> {
        if self.start >= self.end {
            None
        } else {
            let value = self.start;
            self.start = self.start + <<I as IndexBase>::Type as num_traits::identities::One>::one();
            Some(Into::into(value))
        }
    }
}

/// Matrices indexed by new types
pub mod matrix;
/// Bijective mappings between new type indices
pub mod bijection;
/// Surjective mapping between new type indices
pub mod surjection;

#[cfg(test)]
mod tests {
    use itertools::Itertools;
    use crate::strong::{Index, IndexBase};

    #[derive(IndexBase, Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Foo(u8);

    #[test]
    fn index_basics() {
        let a: Foo = 8.into();
        let _b: u8 = a.into();
        // let c: usize = a.into(); compilation failure
        let _d = Foo::from(3);

        assert_eq!(Foo::range(3).count(), 3);
        itertools::assert_equal(
            Foo::range(12),
            (0..12).map_into::<Foo>()
        );
    }
}
