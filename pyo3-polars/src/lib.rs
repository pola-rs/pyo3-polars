//! This crate offers [`PySeries`], [`PyDataFrame`] and [`PyAnyValue`] which are simple wrapper around `Series`, `DataFrame` and `PyAnyValue`. The
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
//! fn expression_lib(_py: Python, m: &PyModule) -> PyResult<()> {
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
use polars::prelude::*;
use pyo3::{FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python, ToPyObject, types::{PyDelta, PyNone, PyDate, PyDateTime, PyTime}, exceptions::PyTypeError};

#[cfg(feature = "lazy")]
use {polars_lazy::frame::LazyFrame, polars_plan::logical_plan::LogicalPlan};

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`Series`] that can be converted to and from python with `pyo3`.
pub struct PySeries(pub Series);

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`DataFrame`] that can be converted to and from python with `pyo3`.
pub struct PyDataFrame(pub DataFrame);

#[repr(transparent)]
#[derive(Debug, Clone)]
/// A wrapper around a [`AnyValue`] that can be converted to and from python with `pyo3`.
pub struct PyAnyValue<'a>(pub AnyValue<'a>);

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

impl<'a> From<PyAnyValue<'a>> for AnyValue<'a> {
    fn from(value: PyAnyValue<'a>) -> Self {
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

impl<'a> AsRef<AnyValue<'a>> for PyAnyValue<'a> {
    fn as_ref(&self) -> &AnyValue<'a> {
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

impl<'a> FromPyObject<'a> for PyAnyValue<'a> {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let object_type = ob
            .getattr("__class__")?
            .getattr("__name__")?
            .extract::<&str>()?;

        match object_type {
            "float" => Ok(PyAnyValue(AnyValue::Float64(ob.extract::<f64>()?))),
            "int" => Ok(PyAnyValue(AnyValue::Int64(ob.extract::<i64>()?))),
            "str" => Ok(PyAnyValue(AnyValue::Utf8(ob.extract::<&str>()?))),
            "bool" => Ok(PyAnyValue(AnyValue::Boolean(ob.extract::<bool>()?))),
            "datetime" => {
                let timestamp = (ob.call_method0("timestamp")?.extract::<f64>()? * 1_000.0) as i64;
                Ok(PyAnyValue(AnyValue::Datetime(
                    timestamp,
                    TimeUnit::Milliseconds,
                    &None,
                )))
            }
            "date" => {
                let days: Result<i32, PyErr> = Python::with_gil(|py| {
                    let datetime = py.import("datetime")?;

                    let epoch = datetime.call_method1("date", (1970, 1, 1))?;

                    let days = ob
                        .call_method1("__sub__", (epoch,))?
                        .getattr("days")?
                        .extract::<i32>()?;

                    Ok(days)
                });
                Ok(PyAnyValue(AnyValue::Date(days?)))
            }
            "timedelta" => {
                let seconds =
                    (ob.call_method0("total_seconds")?.extract::<f64>()? * 1_000.0) as i64;
                Ok(PyAnyValue(AnyValue::Duration(
                    seconds,
                    TimeUnit::Milliseconds,
                )))
            }
            "time" => {
                let hours = ob.getattr("hour")?.extract::<i64>()?;
                let minutes = ob.getattr("minute")?.extract::<i64>()?;
                let seconds = ob.getattr("second")?.extract::<i64>()?;
                let microseconds = ob.getattr("microsecond")?.extract::<i64>()?;

                Ok(PyAnyValue(AnyValue::Time(
                    (hours * 3_600_000_000_000)
                        + (minutes * 60_000_000_000)
                        + (seconds * 1_000_000_000)
                        + (microseconds * 1_000),
                )))
            }
            "Series" => Ok(PyAnyValue(AnyValue::List(ob.extract::<PySeries>()?.0))),
            "bytes" => Ok(PyAnyValue(AnyValue::Binary(ob.extract::<&[u8]>()?))),
            "NoneType" => Ok(PyAnyValue(AnyValue::Null)),
            _ => Err(PyTypeError::new_err(format!(
                "'{}' object cannot be interpreted",
                object_type
            ))),
        }
    }
}

#[cfg(feature = "lazy")]
impl<'a> FromPyObject<'a> for PyLazyFrame {
    fn extract(ob: &'a PyAny) -> PyResult<Self> {
        let s = ob.call_method0("__getstate__")?.extract::<Vec<u8>>()?;
        let lp: LogicalPlan = ciborium::de::from_reader(&*s).map_err(
            |e| PyPolarsErr::Other(
                format!("Error when deserializing LazyFrame. This may be due to mismatched polars versions. {}", e)
            )
        )?;
        Ok(PyLazyFrame(LazyFrame::from(lp)))
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

macro_rules! convert_duration (
    ($py:expr, $difference:expr, $second_factor:literal) => {
        {
            let days = $difference / ($second_factor * 86_400);
            let remaining_after_days = $difference % ($second_factor * 86_400);
            let seconds = remaining_after_days / $second_factor;
            let remaining_after_seconds = remaining_after_days % $second_factor;
            let microseconds = remaining_after_seconds * (1_000_000 / $second_factor);

            PyDelta::new(
                $py,
                i32::try_from(days).unwrap(),
                i32::try_from(seconds).unwrap(),
                i32::try_from(microseconds).unwrap(),
                false,
            )
            .unwrap()
            .into_py($py)
        }
    }
);

impl IntoPy<PyObject> for PyAnyValue<'_> {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self.0 {
            AnyValue::Binary(val) => val.into_py(py),
            AnyValue::Null => PyNone::get(py).into_py(py),
            AnyValue::Boolean(val) => val.into_py(py),
            AnyValue::Utf8(val) => val.into_py(py),
            AnyValue::UInt8(val) => val.into_py(py),
            AnyValue::UInt16(val) => val.into_py(py),
            AnyValue::UInt32(val) => val.into_py(py),
            AnyValue::UInt64(val) => val.into_py(py),
            AnyValue::Int8(val) => val.into_py(py),
            AnyValue::Int16(val) => val.into_py(py),
            AnyValue::Int32(val) => val.into_py(py),
            AnyValue::Int64(val) => val.into_py(py),
            AnyValue::Float32(val) => val.into_py(py),
            AnyValue::Float64(val) => val.into_py(py),
            AnyValue::Date(days) => PyDate::from_timestamp(py, (days * 86_400).into())
                .unwrap()
                .into_py(py),
            // The timezone is ignored - This may lead to wrong conversions
            AnyValue::Datetime(time, unit, _timezone) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    PyDateTime::from_timestamp(py, (time / 1_000_000_000) as f64, None)
                        .unwrap()
                        .into_py(py)
                }
            },
            AnyValue::Duration(difference, unit) => match unit {
                polars::prelude::TimeUnit::Milliseconds => {
                    convert_duration!(py, difference, 1_000)
                }
                polars::prelude::TimeUnit::Microseconds => {
                    convert_duration!(py, difference, 1_000_000)
                }
                polars::prelude::TimeUnit::Nanoseconds => {
                    convert_duration!(py, difference, 1_000_000_000)
                }
            },
            AnyValue::Time(nanoseconds) => {
                let hours = nanoseconds / 3_600_000_000_000;
                let remaining_after_hours = nanoseconds % 3_600_000_000_000;
                let minutes = remaining_after_hours / 60_000_000_000;
                let remaining_after_minutes = remaining_after_hours % 60_000_000_000;
                let seconds = remaining_after_minutes / 1_000_000_000;
                let remaining_after_seconds = remaining_after_minutes % 1_000_000_000;
                let microseconds = remaining_after_seconds / 1_000;

                PyTime::new(
                    py,
                    u8::try_from(hours).unwrap(),
                    u8::try_from(minutes).unwrap(),
                    u8::try_from(seconds).unwrap(),
                    u32::try_from(microseconds).unwrap(),
                    None,
                )
                .unwrap()
                .into_py(py)
            }
            AnyValue::List(val) => PySeries(val).into_py(py),
            AnyValue::Utf8Owned(val) => val.into_py(py),
            AnyValue::BinaryOwned(val) => val.into_py(py),
        }
    }
}

#[cfg(feature = "lazy")]
impl IntoPy<PyObject> for PyLazyFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = py.import("polars").expect("polars not installed");
        let cls = polars.getattr("LazyFrame").unwrap();
        let instance = cls.call_method1("__new__", (cls,)).unwrap();
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.0.logical_plan, &mut writer).unwrap();

        instance.call_method1("__setstate__", (&*writer,)).unwrap();
        instance.into_py(py)
    }
}
