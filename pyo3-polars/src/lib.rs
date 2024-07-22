//! This crate offers a [`PySeries`] and a [`PyDataFrame`] which are simple wrapper around `Series` and `DataFrame`. The
//! advantage of these wrappers is that they can be converted to and from python as they implement `FromPyObject` and `IntoPy`.
//!
//! # Example
//!
//! From `src/lib.rs`.
//! ```rust
//! # use polars::prelude::*;
//! # use pyo3::prelude::*;
//! # use pyo3_polars::PyDataFrame;
//!
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
//! fn expression_lib(_py: Python, m: &Bound<PyModule>) -> PyResult<()> {
//!     m.add_function(wrap_pyfunction!(my_cool_function, m)?)?;
//!     Ok(())
//! }
//! ```
//!
//! Compile your crate with `maturin` and then import from python.
//!
//! From `my_python_file.py`.
//! ```python
//! from expression_lib import my_cool_function
//!
//! df = pl.DataFrame({
//!     "foo": [1, 2, None],
//!     "bar": ["a", None, "c"],
//! })
//! out_df = my_cool_function(df)
//! ```
#[cfg(feature = "derive")]
pub mod derive;
pub mod error;
#[cfg(feature = "derive")]
pub mod export;
mod ffi;

use crate::error::PyPolarsErr;
use crate::ffi::to_py::to_py_array;
use polars::export::arrow;
use polars::prelude::*;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyDict;
#[cfg(feature = "lazy")]
use {polars_lazy::frame::LazyFrame, polars_plan::plans::DslPlan};

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`Series`] that can be converted to and from python with `pyo3`.
pub struct PySeries(pub Series);

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
pub struct PyDataFrame(pub DataFrame);

#[cfg(feature = "lazy")]
#[repr(transparent)]
#[derive(Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
/// # Warning
/// If the [`LazyFrame`] contains in memory data,
/// such as a [`DataFrame`] this will be serialized/deserialized.
///
/// It is recommended to only have `LazyFrame`s that scan data
/// from disk
pub struct PyLazyFrame(pub LazyFrame);

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

#[cfg(feature = "lazy")]
impl From<PyLazyFrame> for LazyFrame {
    fn from(value: PyLazyFrame) -> Self {
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

#[cfg(feature = "lazy")]
impl AsRef<LazyFrame> for PyLazyFrame {
    fn as_ref(&self) -> &LazyFrame {
        &self.0
    }
}

impl<'a> FromPyObject<'a> for PySeries {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let ob = ob.call_method0("rechunk")?;

        let name = ob.getattr("name")?;
        let py_name = name.str()?;
        let name = py_name.to_cow()?;

        let kwargs = PyDict::new_bound(ob.py());
        if let Ok(compat_level) = ob.call_method0("_newest_compat_level") {
            let compat_level = compat_level.extract().unwrap();
            let compat_level =
                CompatLevel::with_level(compat_level).unwrap_or(CompatLevel::newest());
            kwargs.set_item("compat_level", compat_level.get_level())?;
        }
        let arr = ob.call_method("to_arrow", (), Some(&kwargs))?;
        let arr = ffi::to_rust::array_to_rust(&arr)?;
        Ok(PySeries(
            Series::try_from((&*name, arr)).map_err(PyPolarsErr::from)?,
        ))
    }
}

impl<'a> FromPyObject<'a> for PyDataFrame {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let series = ob.call_method0("get_columns")?;
        let n = ob.getattr("width")?.extract::<usize>()?;
        let mut columns = Vec::with_capacity(n);
        for pyseries in series.iter()? {
            let pyseries = pyseries?;
            let s = pyseries.extract::<PySeries>()?.0;
            columns.push(s);
        }
        unsafe { Ok(PyDataFrame(DataFrame::new_no_checks(columns))) }
    }
}

#[cfg(feature = "lazy")]
impl<'a> FromPyObject<'a> for PyLazyFrame {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.call_method0("__getstate__")?.extract::<Vec<u8>>()?;
        let lp: DslPlan = ciborium::de::from_reader(&*s).map_err(
            |e| PyPolarsErr::Other(
                format!("Error when deserializing LazyFrame. This may be due to mismatched polars versions. {}", e)
            )
        )?;
        Ok(PyLazyFrame(LazyFrame::from(lp)))
    }
}

impl IntoPy<PyObject> for PySeries {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = py.import_bound("polars").expect("polars not installed");
        let s = polars.getattr("Series").unwrap();
        match s
            .getattr("_import_arrow_from_c")
            .or_else(|_| s.getattr("_import_from_c"))
        {
            // Go via polars
            Ok(import_arrow_from_c) => {
                // Get supported compatibility level
                let compat_level = CompatLevel::with_level(
                    s.getattr("_newest_compat_level")
                        .map_or(1, |newest_compat_level| {
                            newest_compat_level.call0().unwrap().extract().unwrap()
                        }),
                )
                .unwrap_or(CompatLevel::newest());
                // Prepare pointers on the heap.
                let mut chunk_ptrs = Vec::with_capacity(self.0.n_chunks());
                for i in 0..self.0.n_chunks() {
                    let array = self.0.to_arrow(i, compat_level);
                    let schema = Box::leak(Box::new(arrow::ffi::export_field_to_c(
                        &ArrowField::new("", array.data_type().clone(), true),
                    )));
                    let array = Box::leak(Box::new(arrow::ffi::export_array_to_c(array.clone())));

                    let schema_ptr: *const arrow::ffi::ArrowSchema = &*schema;
                    let array_ptr: *const arrow::ffi::ArrowArray = &*array;
                    chunk_ptrs.push((schema_ptr as Py_uintptr_t, array_ptr as Py_uintptr_t))
                }
                // Somehow we need to clone the Vec, because pyo3 doesn't accept a slice here.
                let pyseries = import_arrow_from_c
                    .call1((self.0.name(), chunk_ptrs.clone()))
                    .unwrap();
                // Deallocate boxes
                for (schema_ptr, array_ptr) in chunk_ptrs {
                    let schema_ptr = schema_ptr as *mut arrow::ffi::ArrowSchema;
                    let array_ptr = array_ptr as *mut arrow::ffi::ArrowArray;
                    unsafe {
                        // We can drop both because the `schema` isn't read in an owned matter on the other side.
                        let _ = Box::from_raw(schema_ptr);

                        // The array is `ptr::read_unaligned` so there are two owners.
                        // We drop the box, and forget the content so the other process is the owner.
                        let array = Box::from_raw(array_ptr);
                        // We must forget because the other process will call the release callback.
                        let array = *array;
                        std::mem::forget(array);
                    }
                }

                pyseries.to_object(py)
            }
            // Go via pyarrow
            Err(_) => {
                let s = self.0.rechunk();
                let name = s.name();
                let arr = s.to_arrow(0, CompatLevel::oldest());
                let pyarrow = py.import_bound("pyarrow").expect("pyarrow not installed");

                let arg = to_py_array(arr, py, pyarrow).unwrap();
                let s = polars.call_method1("from_arrow", (arg,)).unwrap();
                let s = s.call_method1("rename", (name,)).unwrap();
                s.to_object(py)
            }
        }
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

        let polars = py.import_bound("polars").expect("polars not installed");
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}

#[cfg(feature = "lazy")]
impl IntoPy<PyObject> for PyLazyFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = py.import_bound("polars").expect("polars not installed");
        let cls = polars.getattr("LazyFrame").unwrap();
        let instance = cls.call_method1("__new__", (&cls,)).unwrap();
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.0.logical_plan, &mut writer).unwrap();

        instance.call_method1("__setstate__", (&*writer,)).unwrap();
        instance.into_py(py)
    }
}
