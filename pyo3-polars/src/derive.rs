use polars::prelude::PolarsError;
pub use pyo3_polars_derive::polars_expr;
pub use serde_json;
pub use serde_json::{Map, Value};
use std::cell::RefCell;
use std::ffi::CString;
pub type Kwargs = serde_json::Map<String, Value>;

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

pub unsafe fn _parse_kwargs(kwargs: &[u8]) -> Kwargs {
    let kwargs = std::str::from_utf8_unchecked(kwargs);
    let value = serde_json::from_str(kwargs).unwrap();
    if let Value::Object(kwargs) = value {
        return kwargs;
    } else {
        panic!("expected kwargs dictionary")
    }
}

pub fn _update_last_error(err: PolarsError) {
    let msg = format!("{}", err);
    let msg = CString::new(msg).unwrap();
    LAST_ERROR.with(|prev| *prev.borrow_mut() = msg)
}

#[no_mangle]
pub unsafe extern "C" fn get_last_error_message() -> *const std::os::raw::c_char {
    LAST_ERROR.with(|prev| prev.borrow_mut().as_ptr())
}
