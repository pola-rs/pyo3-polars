use polars_core::error::PolarsResult;
use pyo3_polars_derive::polars_expr;
use polars_core::prelude::{Series, Field};
use polars_plan::dsl::FieldsMapper;

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


fn main() {

}