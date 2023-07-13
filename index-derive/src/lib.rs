extern crate proc_macro;

use quote::quote;
use syn::{parse_macro_input, DeriveInput};

use proc_macro::TokenStream;

#[proc_macro_derive(IndexBase)]
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
            impl IndexBase for #name {
                type Type = #field_type;

                fn get(&self) -> #field_type {
                    self.0
                }
            }

            impl ::core::convert::From<#field_type> for #name {
                fn from(original: #field_type) -> #name {
                    #name(original)
                }
            }

            impl ::core::convert::Into<#field_type> for #name {
                fn into(self) -> #field_type {
                    self.0
                }
            }
        };

        TokenStream::from(expanded)
    } else {
        panic!("Expected a struct for Index impl");
    }
}
