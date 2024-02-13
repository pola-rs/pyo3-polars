//! This crate offers [`PySeries`], [`PyDataFrame`] and [`PyAnyValue`] which are simple wrapper around `Series`, `DataFrame` and `AnyValue`. The
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
mod gil_once_cell;
mod py_modules;
mod utils;

use crate::error::PyPolarsErr;
use crate::ffi::to_py::to_py_array;
use crate::py_modules::SERIES;
use crate::utils::{
    abs_decimal_from_digits, any_values_to_dtype, convert_date, convert_datetime, decimal_to_digits,
};
use polars::export::arrow;
use polars::prelude::*;
use py_modules::UTILS;
use pyo3::ffi::Py_uintptr_t;
use pyo3::types::{PyBool, PyDict, PyFloat, PyList, PySequence, PyString, PyTuple, PyType};
use pyo3::{intern, PyErr};
use pyo3::{FromPyObject, IntoPy, PyAny, PyObject, PyResult, Python, ToPyObject};
use utils::struct_dict;

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
/// A wrapper around [`AnyValue`] that can be converted to and from python with `pyo3`.
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

impl<'a> From<AnyValue<'a>> for PyAnyValue<'a> {
    fn from(value: AnyValue<'a>) -> Self {
        PyAnyValue(value)
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

type TypeObjectPtr = usize;
type InitFn = fn(&PyAny) -> PyResult<PyAnyValue<'_>>;
pub(crate) static LUT: crate::gil_once_cell::GILOnceCell<PlHashMap<TypeObjectPtr, InitFn>> =
    crate::gil_once_cell::GILOnceCell::new();

impl<'s> FromPyObject<'s> for PyAnyValue<'s> {
    fn extract(ob: &'s PyAny) -> PyResult<Self> {
        // conversion functions
        fn get_bool(ob: &PyAny) -> PyResult<PyAnyValue<'_>> {
            Ok(AnyValue::Boolean(ob.extract::<bool>().unwrap()).into())
        }

        fn get_int(ob: &PyAny) -> PyResult<PyAnyValue<'_>> {
            // can overflow
            match ob.extract::<i64>() {
                Ok(v) => Ok(AnyValue::Int64(v).into()),
                Err(_) => Ok(AnyValue::UInt64(ob.extract::<u64>()?).into()),
            }
        }

        fn get_float(ob: &PyAny) -> PyResult<PyAnyValue<'_>> {
            Ok(AnyValue::Float64(ob.extract::<f64>().unwrap()).into())
        }

        fn get_str(ob: &PyAny) -> PyResult<PyAnyValue<'_>> {
            let value = ob.extract::<&str>().unwrap();
            Ok(AnyValue::String(value).into())
        }

        fn get_struct(ob: &PyAny) -> PyResult<PyAnyValue<'_>> {
            let dict = ob.downcast::<PyDict>().unwrap();
            let len = dict.len();
            let mut keys = Vec::with_capacity(len);
            let mut vals = Vec::with_capacity(len);
            for (k, v) in dict.into_iter() {
                let key = k.extract::<&str>()?;
                let val = v.extract::<PyAnyValue>()?.0;
                let dtype = DataType::from(&val);
                keys.push(Field::new(key, dtype));
                vals.push(val)
            }
            Ok(AnyValue::StructOwned(Box::new((vals, keys))).into())
        }

        fn get_list(ob: &PyAny) -> PyResult<PyAnyValue> {
            fn get_list_with_constructor(ob: &PyAny) -> PyResult<PyAnyValue> {
                // Use the dedicated constructor
                // this constructor is able to go via dedicated type constructors
                // so it can be much faster
                Python::with_gil(|py| {
                    let s = SERIES.call1(py, (ob,))?;
                    get_series_el(s.as_ref(py))
                })
            }

            if ob.is_empty()? {
                Ok(AnyValue::List(Series::new_empty("", &DataType::Null)).into())
            } else if ob.is_instance_of::<PyList>() | ob.is_instance_of::<PyTuple>() {
                let list = ob.downcast::<PySequence>().unwrap();

                let mut avs = Vec::with_capacity(25);
                let mut iter = list.iter()?;

                for item in (&mut iter).take(25) {
                    avs.push(item?.extract::<PyAnyValue>()?.0)
                }

                let (dtype, n_types) = any_values_to_dtype(&avs).map_err(PyPolarsErr::from)?;

                // we only take this path if there is no question of the data-type
                if dtype.is_primitive() && n_types == 1 {
                    get_list_with_constructor(ob)
                } else {
                    // push the rest
                    avs.reserve(list.len()?);
                    for item in iter {
                        avs.push(item?.extract::<PyAnyValue>()?.0)
                    }

                    let s = Series::from_any_values_and_dtype("", &avs, &dtype, true)
                        .map_err(PyPolarsErr::from)?;
                    Ok(AnyValue::List(s).into())
                }
            } else {
                // range will take this branch
                get_list_with_constructor(ob)
            }
        }

        fn get_series_el(ob: &PyAny) -> PyResult<PyAnyValue<'static>> {
            let py_pyseries = ob.getattr(intern!(ob.py(), "_s")).unwrap();
            let series = py_pyseries.extract::<PySeries>().unwrap().0;
            Ok(AnyValue::List(series).into())
        }

        fn get_bin(ob: &PyAny) -> PyResult<PyAnyValue> {
            let value = ob.extract::<&[u8]>().unwrap();
            Ok(AnyValue::Binary(value).into())
        }

        fn get_null(_ob: &PyAny) -> PyResult<PyAnyValue> {
            Ok(AnyValue::Null.into())
        }

        fn get_timedelta(ob: &PyAny) -> PyResult<PyAnyValue> {
            Python::with_gil(|py| {
                let td = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_timedelta_to_pl_timedelta"))
                    .unwrap()
                    .call1((ob, intern!(py, "us")))
                    .unwrap();
                let v = td.extract::<i64>().unwrap();
                Ok(AnyValue::Duration(v, TimeUnit::Microseconds).into())
            })
        }

        fn get_time(ob: &PyAny) -> PyResult<PyAnyValue> {
            Python::with_gil(|py| {
                let time = UTILS
                    .as_ref(py)
                    .getattr(intern!(py, "_time_to_pl_time"))
                    .unwrap()
                    .call1((ob,))
                    .unwrap();
                let v = time.extract::<i64>().unwrap();
                Ok(AnyValue::Time(v).into())
            })
        }

        fn get_decimal(ob: &PyAny) -> PyResult<PyAnyValue> {
            let (sign, digits, exp): (i8, Vec<u8>, i32) = ob
                .call_method0(intern!(ob.py(), "as_tuple"))
                .unwrap()
                .extract()
                .unwrap();
            // note: using Vec<u8> is not the most efficient thing here (input is a tuple)
            let (mut v, scale) = abs_decimal_from_digits(digits, exp).ok_or_else(|| {
                PyErr::from(PyPolarsErr::Other(
                    "Decimal is too large to fit in Decimal128".into(),
                ))
            })?;
            if sign > 0 {
                v = -v; // won't overflow since -i128::MAX > i128::MIN
            }
            Ok(AnyValue::Decimal(v, scale).into())
        }

        fn get_object(_ob: &PyAny) -> PyResult<PyAnyValue> {
            // TODO: need help here
            // #[cfg(feature = "object")]
            // {
            //     // this is slow, but hey don't use objects
            //     let v = &ObjectValue { inner: ob.into() };
            //     Ok(AnyValue::ObjectOwned(OwnedObject(v.to_boxed())).into())
            // }
            #[cfg(not(feature = "object"))]
            {
                panic!("activate object")
            }
        }

        // TYPE key
        let type_object_ptr = PyType::as_type_ptr(ob.get_type()) as usize;

        Python::with_gil(|py| {
            LUT.with_gil(py, |lut| {
                // get the conversion function
                let convert_fn = lut.entry(type_object_ptr).or_insert_with(
                    // This only runs if type is not in LUT
                    || {
                        if ob.is_instance_of::<PyBool>() {
                            get_bool
                            // TODO: this heap allocs on failure
                        } else if ob.extract::<i64>().is_ok() || ob.extract::<u64>().is_ok() {
                            get_int
                        } else if ob.is_instance_of::<PyFloat>() {
                            get_float
                        } else if ob.is_instance_of::<PyString>() {
                            get_str
                        } else if ob.is_instance_of::<PyDict>() {
                            get_struct
                        } else if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
                            get_list
                        } else if ob.hasattr(intern!(py, "_s")).unwrap() {
                            get_series_el
                        }
                        // TODO: this heap allocs on failure
                        else if ob.extract::<&'s [u8]>().is_ok() {
                            get_bin
                        } else if ob.is_none() {
                            get_null
                        } else {
                            let type_name = ob.get_type().name().unwrap();
                            match type_name {
                                "datetime" => convert_datetime,
                                "date" => convert_date,
                                "timedelta" => get_timedelta,
                                "time" => get_time,
                                "Decimal" => get_decimal,
                                "range" => get_list,
                                _ => {
                                    // special branch for np.float as this fails isinstance float
                                    if ob.extract::<f64>().is_ok() {
                                        return get_float;
                                    }

                                    // Can't use pyo3::types::PyDateTime with abi3-py37 feature,
                                    // so need this workaround instead of `isinstance(ob, datetime)`.
                                    let bases = ob
                                        .get_type()
                                        .getattr(intern!(py, "__bases__"))
                                        .unwrap()
                                        .iter()
                                        .unwrap();
                                    for base in bases {
                                        let parent_type =
                                            base.unwrap().str().unwrap().to_str().unwrap();
                                        match parent_type {
                                            "<class 'datetime.datetime'>" => {
                                                // `datetime.datetime` is a subclass of `datetime.date`,
                                                // so need to check `datetime.datetime` first
                                                return convert_datetime;
                                            }
                                            "<class 'datetime.date'>" => {
                                                return convert_date;
                                            }
                                            _ => (),
                                        }
                                    }

                                    get_object
                                }
                            }
                        }
                    },
                );

                convert_fn(ob)
            })
        })
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
        let polars = py.import("polars").expect("polars not installed");
        let s = polars.getattr("Series").unwrap();
        match s.getattr("_import_from_c") {
            // Go via polars
            Ok(import_from_c) => {
                // Prepare pointers on the heap.
                let mut chunk_ptrs = Vec::with_capacity(self.0.n_chunks());
                for i in 0..self.0.n_chunks() {
                    let array = self.0.to_arrow(i, true);
                    let schema = Box::leak(Box::new(arrow::ffi::export_field_to_c(
                        &ArrowField::new("", array.data_type().clone(), true),
                    )));
                    let array = Box::leak(Box::new(arrow::ffi::export_array_to_c(array.clone())));

                    let schema_ptr: *const arrow::ffi::ArrowSchema = &*schema;
                    let array_ptr: *const arrow::ffi::ArrowArray = &*array;
                    chunk_ptrs.push((schema_ptr as Py_uintptr_t, array_ptr as Py_uintptr_t))
                }
                // Somehow we need to clone the Vec, because pyo3 doesn't accept a slice here.
                let pyseries = import_from_c
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
                let arr = s.to_arrow(0, false);
                let pyarrow = py.import("pyarrow").expect("pyarrow not installed");

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

        let polars = py.import("polars").expect("polars not installed");
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}

impl IntoPy<PyObject> for PyAnyValue<'_> {
    fn into_py(self, py: Python) -> PyObject {
        let utils = UTILS.as_ref(py);
        match self.0 {
            AnyValue::UInt8(v) => v.into_py(py),
            AnyValue::UInt16(v) => v.into_py(py),
            AnyValue::UInt32(v) => v.into_py(py),
            AnyValue::UInt64(v) => v.into_py(py),
            AnyValue::Int8(v) => v.into_py(py),
            AnyValue::Int16(v) => v.into_py(py),
            AnyValue::Int32(v) => v.into_py(py),
            AnyValue::Int64(v) => v.into_py(py),
            AnyValue::Float32(v) => v.into_py(py),
            AnyValue::Float64(v) => v.into_py(py),
            AnyValue::Null => py.None(),
            AnyValue::Boolean(v) => v.into_py(py),
            AnyValue::String(v) => v.into_py(py),
            AnyValue::StringOwned(v) => v.into_py(py),
            AnyValue::Categorical(idx, rev, arr) | AnyValue::Enum(idx, rev, arr) => {
                let s = if arr.is_null() {
                    rev.get(idx)
                } else {
                    unsafe { arr.deref_unchecked().value(idx as usize) }
                };
                s.into_py(py)
            }
            AnyValue::Date(v) => {
                let convert = utils.getattr(intern!(py, "_to_python_date")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            }
            AnyValue::Datetime(v, time_unit, time_zone) => {
                let convert = utils.getattr(intern!(py, "_to_python_datetime")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert
                    .call1((v, time_unit, time_zone.as_ref().map(|s| s.as_str())))
                    .unwrap()
                    .into_py(py)
            }
            AnyValue::Duration(v, time_unit) => {
                let convert = utils.getattr(intern!(py, "_to_python_timedelta")).unwrap();
                let time_unit = time_unit.to_ascii();
                convert.call1((v, time_unit)).unwrap().into_py(py)
            }
            AnyValue::Time(v) => {
                let convert = utils.getattr(intern!(py, "_to_python_time")).unwrap();
                convert.call1((v,)).unwrap().into_py(py)
            }
            // TODO: need help here
            AnyValue::Array(_v, _) | AnyValue::List(_v) => {
                todo!();
                // PySeries(v).to_list()
            }
            ref av @ AnyValue::Struct(_, _, flds) => struct_dict(py, av._iter_struct_av(), flds),
            AnyValue::StructOwned(payload) => struct_dict(py, payload.0.into_iter(), &payload.1),
            // TODO: Also need help here
            // #[cfg(feature = "object")]
            // AnyValue::Object(v) => {
            //     let object = v.as_any().downcast_ref::<ObjectValue>().unwrap();
            //     object.inner.clone()
            // }
            // #[cfg(feature = "object")]
            // AnyValue::ObjectOwned(v) => {
            //     let object = v.0.as_any().downcast_ref::<ObjectValue>().unwrap();
            //     object.inner.clone()
            // }
            AnyValue::Binary(v) => v.into_py(py),
            AnyValue::BinaryOwned(v) => v.into_py(py),
            AnyValue::Decimal(v, scale) => {
                let convert = utils.getattr(intern!(py, "_to_python_decimal")).unwrap();
                const N: usize = 3;
                let mut buf = [0_u128; N];
                let n_digits = decimal_to_digits(v.abs(), &mut buf);
                let buf = unsafe {
                    std::slice::from_raw_parts(
                        buf.as_slice().as_ptr() as *const u8,
                        N * std::mem::size_of::<u128>(),
                    )
                };
                let digits = PyTuple::new(py, buf.iter().take(n_digits));
                convert
                    .call1((v.is_negative() as u8, digits, n_digits, -(scale as i32)))
                    .unwrap()
                    .into_py(py)
            }
        }
    }
}

impl ToPyObject for PyAnyValue<'_> {
    fn to_object(&self, py: Python) -> PyObject {
        self.clone().into_py(py)
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
