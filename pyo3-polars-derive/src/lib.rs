mod attr;
mod keywords;

use proc_macro::TokenStream;
use quote::quote;
use std::sync::atomic::{AtomicBool, Ordering};
use syn::parse_macro_input;

static INIT: AtomicBool = AtomicBool::new(false);

fn insert_error_function() -> proc_macro2::TokenStream {
    let is_init = INIT.swap(true, Ordering::Relaxed);

    // Only expose the error retrieval function on the first expression.
    if !is_init {
        quote!(
            pub use pyo3_polars::derive::get_last_error_message;
        )
    } else {
        proc_macro2::TokenStream::new()
    }
}

fn create_expression_function(ast: syn::ItemFn) -> proc_macro2::TokenStream {
    let fn_name = &ast.sig.ident;
    let error_msg_fn = insert_error_function();

    quote!(
        use pyo3_polars::export::*;

        #error_msg_fn

        // create the outer public function
        #[no_mangle]
        pub unsafe extern "C" fn #fn_name (
            e: *mut polars_ffi::SeriesExport,
            input_len: usize,
            kwargs_ptr: *const u8,
            kwargs_len: usize,
            return_value: *mut polars_ffi::SeriesExport
        )  {
            let inputs = polars_ffi::import_series_buffer(e, input_len).unwrap();

            let kwargs = std::slice::from_raw_parts(kwargs_ptr, kwargs_len);

            let kwargs = if kwargs.is_empty() {
                ::std::option::Option::None
            } else {
                match pyo3_polars::derive::_parse_kwargs(kwargs)  {
                    Ok(value) => Some(value),
                    Err(err) => {
                        pyo3_polars::derive::_update_last_error(err);
                        return;
                    }
                }
            };

            // define the function
            #ast

            // call the function
            let result: PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs, kwargs);
            match result {
                Ok(out) => {
                    // Update return value.
                    *return_value =  polars_ffi::export_series(&out);
                },
                Err(err) => {
                    // Set latest error, but leave return value in empty state.
                    pyo3_polars::derive::_update_last_error(err);
                }
            }
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
    dtype_fn_name: &syn::Ident,
) -> proc_macro2::TokenStream {
    let map_field_name = get_field_name(fn_name);
    let inputs = get_inputs();

    quote! (
        #[no_mangle]
        pub unsafe extern "C" fn #map_field_name(
            field: *mut polars_core::export::arrow::ffi::ArrowSchema,
            len: usize,
            return_value: *mut polars_core::export::arrow::ffi::ArrowSchema,
        ) {
            #inputs;

            let result = #dtype_fn_name(&inputs);

            match result {
                Ok(out) => {
                    let out = polars_core::export::arrow::ffi::export_field_to_c(&out.to_arrow());
                    *return_value = out;
                },
                Err(err) => {
                    // Set latest error, but leave return value in empty state.
                    pyo3_polars::derive::_update_last_error(err);
                }
            }
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
        pub unsafe extern "C" fn #map_field_name(
            field: *mut polars_core::export::arrow::ffi::ArrowSchema,
            len: usize,
            return_value: *mut polars_core::export::arrow::ffi::ArrowSchema
        ) {
            #inputs

            let mapper = polars_plan::dsl::FieldsMapper::new(&inputs);
            let dtype = polars_core::datatypes::DataType::#dtype;
            let out = mapper.with_dtype(dtype).unwrap();
            let out = polars_core::export::arrow::ffi::export_field_to_c(&out.to_arrow());
            *return_value = out;
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
