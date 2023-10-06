pub use pyo3_polars_derive::polars_expr;
pub use serde_json;
pub use serde_json::{Map, Value};
pub type Kwargs = serde_json::Map<String, Value>;

pub unsafe fn parse_kwargs(kwargs: &[u8]) -> Kwargs {
    let kwargs = std::str::from_utf8_unchecked(kwargs);
    let value = serde_json::from_str(kwargs).unwrap();
    if let Value::Object(kwargs) = value {
        return kwargs;
    } else {
        panic!("expected kwargs dictionary")
    }
}
