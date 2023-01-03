mod parallel_jaccard_mod;

use pyo3::prelude::*;
use pyo3_polars::{
    PyDataFrame
};
use pyo3_polars::error::PyPolarsErr;
use polars::prelude::*;


#[pyfunction]
fn parallel_jaccard(pydf: PyDataFrame, col_a: &str, col_b: &str) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    let df = parallel_jaccard_mod::parallel_jaccard(df, col_a, col_b).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))
}

/// A Python module implemented in Rust.
#[pymodule]
fn extend_polars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_jaccard, m)?)?;
    Ok(())
}