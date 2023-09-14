use polars::prelude::*;
use pyo3_polars::derive::polars_expr;

fn pig_latin_str(value: &str) -> String {
    if let Some(first_char) = value.chars().nth(0) {
        format!("{}{}ay", &value[1..], first_char)
    } else {
        value.into()
    }
}

#[polars_expr(output_type=Utf8)]
fn pig_latinnify(inputs: &[Series]) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;

    let out: Utf8Chunked = ca.apply_generic(|opt_v| {
        opt_v.map(pig_latin_str)
    });

    Ok(out.into_series())
}