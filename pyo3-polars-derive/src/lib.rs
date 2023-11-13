mod attr;
mod keywords;

use proc_macro::TokenStream;
use quote::quote;
use std::panic::UnwindSafe;
use std::sync::atomic::{AtomicBool, Ordering};
use syn::{parse_macro_input, FnArg};

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

fn quote_call_kwargs(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(

            let kwargs = std::slice::from_raw_parts(kwargs_ptr, kwargs_len);

            let kwargs = match pyo3_polars::derive::_parse_kwargs(kwargs)  {
                    Ok(value) => value,
                    Err(err) => {
                        pyo3_polars::derive::_update_last_error(err);
                        return;
                    }
            };

            // define the function
            #ast

            // call the function
        let result: PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs, kwargs);

    )
}

fn quote_call_no_kwargs(ast: &syn::ItemFn, fn_name: &syn::Ident) -> proc_macro2::TokenStream {
    quote!(
            // define the function
            #ast
            // call the function
            let result: PolarsResult<polars_core::prelude::Series> = #fn_name(&inputs);
    )
}

fn quote_process_results() -> proc_macro2::TokenStream {
    quote!(match result {
        Ok(out) => {
            // Update return value.
            *return_value = polars_ffi::export_series(&out);
        }
        Err(err) => {
            // Set latest error, but leave return value in empty state.
            pyo3_polars::derive::_update_last_error(err);
        }
    })
}

struct CatchPanic<T>(pub T);
impl<T> UnwindSafe for CatchPanic<T> {}

fn create_expression_function(ast: syn::ItemFn) -> proc_macro2::TokenStream {
    // count how often the user define a kwargs argument.
    let n_kwargs = ast
        .sig
        .inputs
        .iter()
        .filter(|fn_arg| {
            if let FnArg::Typed(pat) = fn_arg {
                if let syn::Pat::Ident(pat) = pat.pat.as_ref() {
                    pat.ident.to_string() == "kwargs"
                } else {
                    false
                }
            } else {
                true
            }
        })
        .count();

    let fn_name = &ast.sig.ident;
    let error_msg_fn = insert_error_function();

    let quote_call = match n_kwargs {
        0 => quote_call_no_kwargs(&ast, fn_name),
        1 => quote_call_kwargs(&ast, fn_name),
        _ => unreachable!(), // arguments are unique
    };
    let quote_process_result = quote_process_results();

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
            let panic_result = std::panic::catch_unwind(move || {
                let inputs = polars_ffi::import_series_buffer(e, input_len).unwrap();

                #quote_call

                #quote_process_result
                ()
            });

            if panic_result.is_err() {
                // Set latest to panic and nullify return value;
                *return_value = polars_ffi::SeriesExport::empty();
                pyo3_polars::derive::_set_panic();
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
            let panic_result = std::panic::catch_unwind(move || {
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
            });

            if panic_result.is_err() {
                // Set latest to panic and nullify return value;
                *return_value = polars_core::export::arrow::ffi::ArrowSchema::empty();
                pyo3_polars::derive::_set_panic();
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
