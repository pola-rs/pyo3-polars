use polars_core::error::PolarsResult;
use pyo3_polars_derive::polars_expr;
use polars_core::prelude::{Series};

#[polars_expr(output_type=Int32)]
fn horizontal_product(series: &[Series]) -> PolarsResult<Series> {
    let mut acc = series[0].clone();
    for s in &series[1..] {
        acc = &acc * s
    }
    Ok(acc)
}


fn main() {

}
