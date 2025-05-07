mod parallel_jaccard_mod;

use polars::prelude::*;
use polars_lazy::frame::IntoLazy;
use polars_lazy::prelude::{Expr, LazyFrame};
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PolarsAllocator, PyDataFrame, PyExpr, PyLazyFrame};

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[pyfunction]
fn parallel_jaccard(pydf: PyDataFrame, col_a: &str, col_b: &str) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    let df = parallel_jaccard_mod::parallel_jaccard(df, col_a, col_b).map_err(PyPolarsErr::from)?;
    Ok(PyDataFrame(df))
}

#[pyfunction]
fn debug(pydf: PyDataFrame) -> PyResult<PyDataFrame> {
    let df: DataFrame = pydf.into();
    dbg!(&df);
    Ok(PyDataFrame(df))
}

#[pyfunction]
fn expr_debug(pyexpr: PyExpr) -> PyResult<PyExpr> {
    let expr: Expr = pyexpr.0;
    dbg!(&expr);
    Ok(PyExpr(expr))
}

#[pyfunction]
fn lazy_debug(pydf: PyLazyFrame) -> PyResult<PyLazyFrame> {
    let lf: LazyFrame = pydf.into();
    dbg!(&lf.describe_plan());
    Ok(PyLazyFrame(lf))
}

#[pyfunction]
fn lazy_parallel_jaccard(pydf: PyLazyFrame, col_a: &str, col_b: &str) -> PyResult<PyLazyFrame> {
    let df: LazyFrame = pydf.into();
    dbg!(&df.describe_plan());
    let df = parallel_jaccard_mod::parallel_jaccard(df.collect().unwrap(), col_a, col_b)
        .map_err(PyPolarsErr::from)?;
    Ok(PyLazyFrame(df.lazy()))
}

/// A Python module implemented in Rust.
#[pymodule]
fn extend_polars(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parallel_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_parallel_jaccard, m)?)?;
    m.add_function(wrap_pyfunction!(debug, m)?)?;
    m.add_function(wrap_pyfunction!(expr_debug, m)?)?;
    m.add_function(wrap_pyfunction!(lazy_debug, m)?)?;
    Ok(())
}
