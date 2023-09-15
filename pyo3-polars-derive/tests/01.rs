use polars_core::error::PolarsResult;
use polars_core::prelude::{Field, Series};
use polars_plan::dsl::FieldsMapper;
use pyo3_polars_derive::polars_expr;

fn horizontal_product_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_supertype()
}

#[polars_expr(type_func=horizontal_product_output)]
fn horizontal_product(series: &[Series]) -> PolarsResult<Series> {
    let mut acc = series[0].clone();
    for s in &series[1..] {
        acc = &acc * s
    }
    Ok(acc)
}

fn main() {}
