mod attr;
mod keywords;

use proc_macro::TokenStream;
use quote::quote;
use syn::parse_macro_input;

fn create_expression_function(ast: syn::ItemFn) -> proc_macro2::TokenStream {
    let fn_name = &ast.sig.ident;

    quote!(
        use pyo3_polars::export::*;
        // create the outer public function
        #[no_mangle]
        pub unsafe extern "C" fn #fn_name (e: *mut polars_ffi::SeriesExport, len: usize) -> polars_ffi::SeriesExport {
            let inputs = polars_ffi::import_series_buffer(e, len).unwrap();

            // define the function
            #ast

            // call the function
            let output: polars_core::prelude::Series = #fn_name(&inputs).unwrap();
            let out = polars_ffi::export_series(&output);
            out
        }
    )
}

fn get_field_name(fn_name: &syn::Ident) -> syn::Ident {
    syn::Ident::new(&format!("__polars_field_{}", fn_name), fn_name.span())
}

fn get_inputs() -> proc_macro2::TokenStream {
    quote!(
             let inputs = std::slice::from_raw_parts(field, len);
             let inputs = inputs.iter().map(|field| {
                 let field = polars_core::export::arrow::ffi::import_field_from_c(field).unwrap();
                 let out = polars_core::prelude::Field::from(&field);
                 out
             }).collect::<Vec<_>>();
    )
}

fn create_field_function(
    fn_name: &syn::Ident, 
    dtype_fn_name: &syn::Ident
) -> proc_macro2::TokenStream {
    let map_field_name = get_field_name(fn_name);
    let inputs = get_inputs();

    quote! (
        #[no_mangle]
        pub unsafe extern "C" fn #map_field_name(field: *mut polars_core::export::arrow::ffi::ArrowSchema, len: usize) -> polars_core::export::arrow::ffi::ArrowSchema {
            #inputs;
            let out = #dtype_fn_name(&inputs).unwrap();
            polars_core::export::arrow::ffi::export_field_to_c(&out.to_arrow())
        }
    )
}

fn create_field_function_from_with_dtype(
    fn_name: &syn::Ident,
    dtype: syn::Ident,
) -> proc_macro2::TokenStream {
    let map_field_name = get_field_name(fn_name);
    let inputs = get_inputs();

    quote! (
        #[no_mangle]
        pub unsafe extern "C" fn #map_field_name(field: *mut polars_core::export::arrow::ffi::ArrowSchema, len: usize) -> polars_core::export::arrow::ffi::ArrowSchema {
            #inputs

            let mapper = polars_plan::dsl::FieldsMapper::new(&inputs);
            let dtype = polars_core::datatypes::DataType::#dtype;
            let out = mapper.with_dtype(dtype).unwrap();
            polars_core::export::arrow::ffi::export_field_to_c(&out.to_arrow())
        }
    )
}

#[proc_macro_attribute]
pub fn polars_expr(attr: TokenStream, input: TokenStream) -> TokenStream {
    let ast = parse_macro_input!(input as syn::ItemFn);

    let options = parse_macro_input!(attr as attr::ExprsFunctionOptions);
    let expanded_field_fn = if let Some(fn_name) = options.output_type_fn {
        create_field_function(&ast.sig.ident, &fn_name)
    } else if let Some(dtype) = options.output_dtype {
        create_field_function_from_with_dtype(&ast.sig.ident, dtype)
    } else {
        panic!("didn't understand polars_expr attribute")
    };

    let expanded_expr = create_expression_function(ast);
    let expanded = quote!(
        #expanded_field_fn

        #expanded_expr
    );
    TokenStream::from(expanded)
}
