use polars::prelude::*;
use polars_plan::dsl::FieldsMapper;
use pyo3_polars::derive::polars_expr;
use serde::Deserialize;
use std::fmt::Write;

#[derive(Deserialize)]
struct PigLatinKwargs {
    capitalize: bool,
}

fn pig_latin_str(value: &str, capitalize: bool, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        if capitalize {
            for c in value.chars().skip(1).map(|char| char.to_uppercase()) {
                write!(output, "{c}").unwrap()
            }
            write!(output, "AY").unwrap()
        } else {
            let offset = first_char.len_utf8();
            write!(output, "{}{}ay", &value[offset..], first_char).unwrap()
        }
    }
}

#[polars_expr(output_type=Utf8)]
fn pig_latinnify(inputs: &[Series], kwargs: PigLatinKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].utf8()?;
    let out: Utf8Chunked =
        ca.apply_to_buffer(|value, output| pig_latin_str(value, kwargs.capitalize, output));
    Ok(out.into_series())
}

#[polars_expr(output_type=Float64)]
fn jaccard_similarity(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].list()?;
    let b = inputs[1].list()?;
    crate::distances::naive_jaccard_sim(a, b).map(|ca| ca.into_series())
}

#[polars_expr(output_type=Float64)]
fn hamming_distance(inputs: &[Series]) -> PolarsResult<Series> {
    let a = inputs[0].utf8()?;
    let b = inputs[1].utf8()?;
    let out: UInt32Chunked =
        arity::binary_elementwise_values(a, b, crate::distances::naive_hamming_dist);
    Ok(out.into_series())
}

fn haversine_output(input_fields: &[Field]) -> PolarsResult<Field> {
    FieldsMapper::new(input_fields).map_to_float_dtype()
}

#[polars_expr(output_type_func=haversine_output)]
fn haversine(inputs: &[Series]) -> PolarsResult<Series> {
    let out = match inputs[0].dtype() {
        DataType::Float32 => {
            let start_lat = inputs[0].f32().unwrap();
            let start_long = inputs[1].f32().unwrap();
            let end_lat = inputs[2].f32().unwrap();
            let end_long = inputs[3].f32().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        DataType::Float64 => {
            let start_lat = inputs[0].f64().unwrap();
            let start_long = inputs[1].f64().unwrap();
            let end_lat = inputs[2].f64().unwrap();
            let end_long = inputs[3].f64().unwrap();
            crate::distances::naive_haversine(start_lat, start_long, end_lat, end_long)?
                .into_series()
        }
        _ => unimplemented!(),
    };
    Ok(out)
}

/// The `DefaultKwargs` isn't very ergonomic as it doesn't validate any schema.
/// Provide your own kwargs struct with the proper schema and accept that type
/// in your plugin expression.
#[derive(Deserialize)]
pub struct MyKwargs {
    float_arg: f64,
    integer_arg: i64,
    string_arg: String,
    boolean_arg: bool,
}

/// If you want to accept `kwargs`. You define a `kwargs` argument
/// on the second position in you plugin. You can provide any custom struct that is deserializable
/// with the pickle protocol (on the rust side).
#[polars_expr(output_type=Utf8)]
fn append_kwargs(input: &[Series], kwargs: MyKwargs) -> PolarsResult<Series> {
    let input = &input[0];
    let input = input.cast(&DataType::Utf8)?;
    let ca = input.utf8().unwrap();

    Ok(ca
        .apply_to_buffer(|val, buf| {
            write!(
                buf,
                "{}-{}-{}-{}-{}",
                val, kwargs.float_arg, kwargs.integer_arg, kwargs.string_arg, kwargs.boolean_arg
            )
            .unwrap()
        })
        .into_series())
}

#[polars_expr(output_type=Boolean)]
fn is_leap_year(input: &[Series]) -> PolarsResult<Series> {
    let input = &input[0];
    let ca = input.date()?;

    let out: BooleanChunked = ca
        .as_date_iter()
        .map(|opt_dt| opt_dt.map(|dt| dt.leap_year()))
        .collect_ca(ca.name());

    Ok(out.into_series())
}
