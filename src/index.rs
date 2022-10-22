use num_traits::int::PrimInt;
use num_traits::FromPrimitive;

pub use index_derive::Index;

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
