//! This crate offers a [`PySeries`] and a [`PyDataFrame`] which are simple wrapper around `Series` and `DataFrame`. The
//! advantage of these wrappers is that they can be converted to and from python as they implement `FromPyObject` and `IntoPy`.
//!
//! # Example
//!
//! From `src/lib.rs`.
//! ```rust
//! #[pyfunction]
//! fn my_cool_function(pydf: PyDataFrame) -> PyResult<PyDataFrame> {
//!     let df: DataFrame = pydf.into();
//!     let df = {
//!         // some work on the dataframe here
//!         todo!()
//!     };
//!
//!     // wrap the dataframe and it will be automatically converted to a python polars dataframe
//!     Ok(PyDataFrame(df))
//! }
//!
//! /// A Python module implemented in Rust.
//! #[pymodule]
//! fn extend_polars(_py: Python, m: &PyModule) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(my_cool_function, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! Compile your crate with `maturin` and then import from python.
//!
//! From `my_python_file.py`.
//! ```python
//! from extend_polars import my_cool_function
//!
//! df = pl.DataFrame({
//!     "foo": [1, 2, None],
//!     "bar": ["a", None, "c"],
//! })
//! out_df = my_cool_function(df)
//! ```
pub mod error;
mod ffi;

use crate::error::PyPolarsErr;
use crate::ffi::to_py::to_py_array;
use polars::prelude::*;
use pyo3::{FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python, ToPyObject};

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`Series`] that can be converted to and from python with `pyo3`.
pub struct PySeries(pub Series);

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
pub struct PyDataFrame(pub DataFrame);

impl From<PyDataFrame> for DataFrame {
    fn from(value: PyDataFrame) -> Self {
        value.0
    }
}

impl From<PySeries> for Series {
    fn from(value: PySeries) -> Self {
        value.0
    }
}

impl AsRef<Series> for PySeries {
    fn as_ref(&self) -> &Series {
        &self.0
    }
}

impl AsRef<DataFrame> for PyDataFrame {
    fn as_ref(&self) -> &DataFrame {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PySeries {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let ob = ob.call_method0("rechunk")?;

        let name = ob.getattr("name")?;
        let name = name.str()?.to_str()?;

        let arr = ob.call_method0("to_arrow")?;
        let arr = ffi::to_rust::array_to_rust(arr)?;
        Ok(PySeries(
            Series::try_from((name, arr)).map_err(PyPolarsErr::from)?,
        ))
    }
}

impl<'a> FromPyObject<'a> for PyDataFrame {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let series = ob.call_method0("get_columns")?;
        let n = ob.getattr("width")?.extract::<usize>()?;
        let mut columns = Vec::with_capacity(n);
        for pyseries in series.iter()? {
            let pyseries = pyseries?;
            let s = pyseries.extract::<PySeries>()?.0;
            columns.push(s);
        }
        Ok(PyDataFrame(DataFrame::new_no_checks(columns)))
    }
}

impl IntoPy<PyObject> for PySeries {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let s = self.0.rechunk();
        let name = s.name();
        let arr = s.to_arrow(0);
        let pyarrow = py.import("pyarrow").expect("pyarrow not installed");
        let polars = py.import("polars").expect("polars not installed");

        let arg = to_py_array(arr, py, pyarrow).unwrap();
        let s = polars.call_method1("from_arrow", (arg,)).unwrap();
        let s = s.call_method1("rename", (name,)).unwrap();
        s.to_object(py)
    }
}

impl IntoPy<PyObject> for PyDataFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let pyseries = self
            .0
            .get_columns()
            .iter()
            .map(|s| PySeries(s.clone()).into_py(py))
            .collect::<Vec<_>>();

        let polars = py.import("polars").expect("polars not installed");
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}
