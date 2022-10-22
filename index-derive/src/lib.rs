extern crate proc_macro;

use quote::quote;
use syn::{parse_macro_input, DeriveInput};

use proc_macro::TokenStream;

#[proc_macro_derive(Index)]
pub fn impl_index(input: TokenStream) -> TokenStream {
    // Parse the string representation
    let ast = parse_macro_input!(input as DeriveInput);

    let name = &ast.ident;

    if let syn::Data::Struct(data) = ast.data {
        if data.fields.len() != 1 {
            panic!("Expected exactly one struct field");
        }
        let head_field = data.fields.iter().next().unwrap();
        let field_type = &head_field.ty;

        let expanded = quote! {
            impl Index for #name {
                type Type = #field_type;

                fn get(&self) -> #field_type {
                    self.0
                }
            }
        };

        TokenStream::from(expanded)
    } else {
        panic!("Expected a struct for Index impl");
    }
}
