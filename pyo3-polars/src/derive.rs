use polars::prelude::PolarsError;
use polars_core::error::{to_compute_err, PolarsResult};
pub use pyo3_polars_derive::polars_expr;
use serde::Deserialize;
use std::cell::RefCell;
use std::ffi::CString;

/// Gives the caller extra information on how to execute the expression.
pub use polars_ffi::version_0::CallerContext;

/// A default opaque kwargs type.
pub type DefaultKwargs = serde_pickle::Value;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub unsafe fn _parse_kwargs<'a, T>(kwargs: &'a [u8]) -> PolarsResult<T>
where
    T: Deserialize<'a>,
{
    serde_pickle::from_slice(kwargs, Default::default()).map_err(to_compute_err)
}

pub fn _update_last_error(err: PolarsError) {
    let msg = format!("{}", err);
    let msg = CString::new(msg).unwrap();
    LAST_ERROR.with(|prev| *prev.borrow_mut() = msg)
}

pub fn _set_panic() {
    let msg = format!("PANIC");
    let msg = CString::new(msg).unwrap();
    LAST_ERROR.with(|prev| *prev.borrow_mut() = msg)
}

#[no_mangle]
pub unsafe extern "C" fn _polars_plugin_get_last_error_message() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|prev| prev.borrow_mut().as_ptr())
}

#[no_mangle]
pub unsafe extern "C" fn _polars_plugin_get_version() -> u32 {
    let (major, minor) = polars_ffi::get_version();
    // Stack bits together
    ((major as u32) << 16) + minor as u32
}
