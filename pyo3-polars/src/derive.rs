pub use pyo3_polars_derive::polars_expr;
pub use serde_json;

pub unsafe fn parse_kwargs(kwargs: &[u8]) -> serde_json::Value {
    let kwargs = std::str::from_utf8_unchecked(kwargs);
    serde_json::from_str(kwargs).unwrap()
}
