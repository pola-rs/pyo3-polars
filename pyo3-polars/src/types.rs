use super::*;
use crate::error::PyPolarsErr;
use crate::ffi::to_py::to_py_array;
use polars::export::arrow;
#[cfg(feature = "dtype-categorical")]
use polars_core::datatypes::create_enum_data_type;
use polars_core::datatypes::{CompatLevel, DataType};
use polars_core::prelude::*;
use polars_core::utils::materialize_dyn_int;
#[cfg(feature = "lazy")]
use polars_lazy::frame::LazyFrame;
#[cfg(feature = "lazy")]
use polars_plan::dsl::Expr;
#[cfg(feature = "lazy")]
use polars_plan::plans::DslPlan;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::ffi::Py_uintptr_t;
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use pyo3::types::PyDict;
#[cfg(feature = "dtype-full")]
use pyo3::types::PyList;

#[cfg(feature = "dtype-categorical")]
pub(crate) fn get_series(obj: &Bound<'_, PyAny>) -> PyResult<Series> {
    let s = obj.getattr(intern!(obj.py(), "_s"))?;
    Ok(s.extract::<PySeries>()?.0)
}

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

#[cfg(feature = "lazy")]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyExpr(pub Expr);

#[repr(transparent)]
#[derive(Clone)]
pub struct PySchema(pub SchemaRef);

#[repr(transparent)]
#[derive(Clone)]
pub struct PyDataType(pub DataType);

#[repr(transparent)]
#[derive(Clone, Copy)]
pub struct PyTimeUnit(TimeUnit);

#[repr(transparent)]
#[derive(Clone)]
pub struct PyField(Field);

impl<'py> FromPyObject<'py> for PyField {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let name = ob
            .getattr(intern!(py, "name"))?
            .str()?
            .extract::<PyBackedStr>()?;
        let dtype = ob.getattr(intern!(py, "dtype"))?.extract::<PyDataType>()?;
        Ok(PyField(Field::new(&name, dtype.0)))
    }
}

impl<'py> FromPyObject<'py> for PyTimeUnit {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let parsed = match &*ob.extract::<PyBackedStr>()? {
            "ns" => TimeUnit::Nanoseconds,
            "us" => TimeUnit::Microseconds,
            "ms" => TimeUnit::Milliseconds,
            v => {
                return Err(PyValueError::new_err(format!(
                    "`time_unit` must be one of {{'ns', 'us', 'ms'}}, got {v}",
                )))
            }
        };
        Ok(PyTimeUnit(parsed))
    }
}

impl ToPyObject for PyTimeUnit {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        let time_unit = match self.0 {
            TimeUnit::Nanoseconds => "ns",
            TimeUnit::Microseconds => "us",
            TimeUnit::Milliseconds => "ms",
        };
        time_unit.into_py(py)
    }
}

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

impl From<PySchema> for SchemaRef {
    fn from(value: PySchema) -> Self {
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

impl AsRef<Schema> for PySchema {
    fn as_ref(&self) -> &Schema {
        self.0.as_ref()
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

#[cfg(feature = "lazy")]
impl<'a> FromPyObject<'a> for PyExpr {
    fn extract_bound(ob: &Bound<'a, PyAny>) -> PyResult<Self> {
        let s = ob.call_method0("__getstate__")?.extract::<Vec<u8>>()?;
        let e: Expr = ciborium::de::from_reader(&*s).map_err(
            |e| PyPolarsErr::Other(
                format!("Error when deserializing 'Expr'. This may be due to mismatched polars versions. {}", e)
            )
        )?;
        Ok(PyExpr(e))
    }
}

impl IntoPy<PyObject> for PySeries {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = POLARS.bind(py);
        let s = SERIES.bind(py);
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
                    let schema = Box::new(arrow::ffi::export_field_to_c(&ArrowField::new(
                        "",
                        array.data_type().clone(),
                        true,
                    )));
                    let array = Box::new(arrow::ffi::export_array_to_c(array.clone()));

                    let schema_ptr: *const arrow::ffi::ArrowSchema = Box::leak(schema);
                    let array_ptr: *const arrow::ffi::ArrowArray = Box::leak(array);

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
                        // Read *array as Box::into_inner
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

        let polars = POLARS.bind(py);
        let df_object = polars.call_method1("DataFrame", (pyseries,)).unwrap();
        df_object.into_py(py)
    }
}

#[cfg(feature = "lazy")]
impl IntoPy<PyObject> for PyLazyFrame {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = POLARS.bind(py);
        let cls = polars.getattr("LazyFrame").unwrap();
        let instance = cls.call_method1(intern!(py, "__new__"), (&cls,)).unwrap();
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.0.logical_plan, &mut writer).unwrap();

        instance.call_method1("__setstate__", (&*writer,)).unwrap();
        instance.into_py(py)
    }
}

#[cfg(feature = "lazy")]
impl IntoPy<PyObject> for PyExpr {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let polars = POLARS.bind(py);
        let cls = polars.getattr("Expr").unwrap();
        let instance = cls.call_method1(intern!(py, "__new__"), (&cls,)).unwrap();
        let mut writer: Vec<u8> = vec![];
        ciborium::ser::into_writer(&self.0, &mut writer).unwrap();

        instance.call_method1("__setstate__", (&*writer,)).unwrap();
        instance.into_py(py)
    }
}

#[cfg(feature = "dtype-full")]
pub(crate) fn to_series(py: Python, s: PySeries) -> PyObject {
    let series = SERIES.bind(py);
    let constructor = series
        .getattr(intern!(series.py(), "_from_pyseries"))
        .unwrap();
    constructor.call1((s,)).unwrap().into_py(py)
}

impl ToPyObject for PyDataType {
    fn to_object(&self, py: Python) -> PyObject {
        let pl = POLARS.bind(py);

        match &self.0 {
            DataType::Int8 => {
                let class = pl.getattr(intern!(py, "Int8")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Int16 => {
                let class = pl.getattr(intern!(py, "Int16")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Int32 => {
                let class = pl.getattr(intern!(py, "Int32")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Int64 => {
                let class = pl.getattr(intern!(py, "Int64")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::UInt8 => {
                let class = pl.getattr(intern!(py, "UInt8")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::UInt16 => {
                let class = pl.getattr(intern!(py, "UInt16")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::UInt32 => {
                let class = pl.getattr(intern!(py, "UInt32")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::UInt64 => {
                let class = pl.getattr(intern!(py, "UInt64")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Float32 => {
                let class = pl.getattr(intern!(py, "Float32")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Float64 | DataType::Unknown(UnknownKind::Float) => {
                let class = pl.getattr(intern!(py, "Float64")).unwrap();
                class.call0().unwrap().into()
            }
            #[cfg(feature = "dtype-decimal")]
            DataType::Decimal(precision, scale) => {
                let class = pl.getattr(intern!(py, "Decimal")).unwrap();
                let args = (*precision, *scale);
                class.call1(args).unwrap().into()
            }
            DataType::Boolean => {
                let class = pl.getattr(intern!(py, "Boolean")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::String | DataType::Unknown(UnknownKind::Str) => {
                let class = pl.getattr(intern!(py, "String")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Binary => {
                let class = pl.getattr(intern!(py, "Binary")).unwrap();
                class.call0().unwrap().into()
            }
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, size) => {
                let class = pl.getattr(intern!(py, "Array")).unwrap();
                let inner = PyDataType(*inner.clone()).to_object(py);
                let args = (inner, *size);
                class.call1(args).unwrap().into()
            }
            DataType::List(inner) => {
                let class = pl.getattr(intern!(py, "List")).unwrap();
                let inner = PyDataType(*inner.clone()).to_object(py);
                class.call1((inner,)).unwrap().into()
            }
            DataType::Date => {
                let class = pl.getattr(intern!(py, "Date")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Datetime(tu, tz) => {
                let datetime_class = pl.getattr(intern!(py, "Datetime")).unwrap();
                datetime_class
                    .call1((tu.to_ascii(), tz.clone()))
                    .unwrap()
                    .into()
            }
            DataType::Duration(tu) => {
                let duration_class = pl.getattr(intern!(py, "Duration")).unwrap();
                duration_class.call1((tu.to_ascii(),)).unwrap().into()
            }
            #[cfg(feature = "object")]
            DataType::Object(_, _) => {
                let class = pl.getattr(intern!(py, "Object")).unwrap();
                class.call0().unwrap().into()
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_, ordering) => {
                let class = pl.getattr(intern!(py, "Categorical")).unwrap();
                let ordering = match ordering {
                    CategoricalOrdering::Physical => "physical",
                    CategoricalOrdering::Lexical => "lexical",
                };
                class.call1((ordering,)).unwrap().into()
            }
            #[cfg(feature = "dtype-categorical")]
            DataType::Enum(rev_map, _) => {
                // we should always have an initialized rev_map coming from rust
                let categories = rev_map.as_ref().unwrap().get_categories();
                let class = pl.getattr(intern!(py, "Enum")).unwrap();
                let s = Series::from_arrow("category", categories.clone().boxed()).unwrap();
                let series = to_series(py, s.into());
                return class.call1((series,)).unwrap().into();
            }
            DataType::Time => pl.getattr(intern!(py, "Time")).unwrap().into(),
            #[cfg(feature = "dtype-struct")]
            DataType::Struct(fields) => {
                let field_class = pl.getattr(intern!(py, "Field")).unwrap();
                let iter = fields.iter().map(|fld| {
                    let name = fld.name().as_str();
                    let dtype = PyDataType(fld.data_type().clone()).to_object(py);
                    field_class.call1((name, dtype)).unwrap()
                });
                let fields = PyList::new_bound(py, iter);
                let struct_class = pl.getattr(intern!(py, "Struct")).unwrap();
                struct_class.call1((fields,)).unwrap().into()
            }
            DataType::Null => {
                let class = pl.getattr(intern!(py, "Null")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::Unknown(UnknownKind::Int(v)) => {
                PyDataType(materialize_dyn_int(*v).dtype()).to_object(py)
            }
            DataType::Unknown(_) => {
                let class = pl.getattr(intern!(py, "Unknown")).unwrap();
                class.call0().unwrap().into()
            }
            DataType::BinaryOffset => {
                panic!("this type isn't exposed to python")
            }
        }
    }
}

impl IntoPy<PyObject> for PySchema {
    fn into_py(self, py: Python<'_>) -> PyObject {
        let dict = PyDict::new_bound(py);
        for (k, v) in self.0.iter() {
            dict.set_item(k.as_str(), PyDataType(v.clone())).unwrap();
        }
        dict.into_py(py)
    }
}

impl<'py> FromPyObject<'py> for PyDataType {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = ob.py();
        let type_name = ob.get_type().qualname()?;
        let type_name = type_name.to_cow()?;

        let dtype = match type_name.as_ref() {
            "DataTypeClass" => {
                // just the class, not an object
                let name = ob
                    .getattr(intern!(py, "__name__"))?
                    .str()?
                    .extract::<PyBackedStr>()?;
                match &*name {
                    "Int8" => DataType::Int8,
                    "Int16" => DataType::Int16,
                    "Int32" => DataType::Int32,
                    "Int64" => DataType::Int64,
                    "UInt8" => DataType::UInt8,
                    "UInt16" => DataType::UInt16,
                    "UInt32" => DataType::UInt32,
                    "UInt64" => DataType::UInt64,
                    "Float32" => DataType::Float32,
                    "Float64" => DataType::Float64,
                    "Boolean" => DataType::Boolean,
                    "String" => DataType::String,
                    "Binary" => DataType::Binary,
                    #[cfg(feature = "dtype-categorical")]
                    "Categorical" => DataType::Categorical(None, Default::default()),
                    #[cfg(feature = "dtype-categorical")]
                    "Enum" => DataType::Enum(None, Default::default()),
                    "Date" => DataType::Date,
                    "Time" => DataType::Time,
                    "Datetime" => DataType::Datetime(TimeUnit::Microseconds, None),
                    "Duration" => DataType::Duration(TimeUnit::Microseconds),
                    #[cfg(feature = "dtype-decimal")]
                    "Decimal" => DataType::Decimal(None, None), // "none" scale => "infer"
                    "List" => DataType::List(Box::new(DataType::Null)),
                    #[cfg(feature = "dtype-array")]
                    "Array" => DataType::Array(Box::new(DataType::Null), 0),
                    #[cfg(feature = "dtype-struct")]
                    "Struct" => DataType::Struct(vec![]),
                    "Null" => DataType::Null,
                    #[cfg(feature = "object")]
                    "Object" => todo!(),
                    "Unknown" => DataType::Unknown(Default::default()),
                    dt => {
                        return Err(PyTypeError::new_err(format!(
                            "'{dt}' is not a Polars data type, or the plugin isn't compiled with the right features",
                        )))
                    },
                }
            },
            "Int8" => DataType::Int8,
            "Int16" => DataType::Int16,
            "Int32" => DataType::Int32,
            "Int64" => DataType::Int64,
            "UInt8" => DataType::UInt8,
            "UInt16" => DataType::UInt16,
            "UInt32" => DataType::UInt32,
            "UInt64" => DataType::UInt64,
            "Float32" => DataType::Float32,
            "Float64" => DataType::Float64,
            "Boolean" => DataType::Boolean,
            "String" => DataType::String,
            "Binary" => DataType::Binary,
            #[cfg(feature = "dtype-categorical")]
            "Categorical" => {
                let ordering = ob.getattr(intern!(py, "ordering")).unwrap();

                let ordering = match ordering.extract::<&str>()? {
                    "physical" => CategoricalOrdering::Physical,
                    "lexical" => CategoricalOrdering::Lexical,
                    ordering => PyValueError::new_err(format!("invalid ordering argument: {ordering}"))
                };

                DataType::Categorical(None, ordering)
            },
            #[cfg(feature = "dtype-categorical")]
            "Enum" => {
                let categories = ob.getattr(intern!(py, "categories")).unwrap();
                let s = get_series(&categories.as_borrowed())?;
                let ca = s.str().map_err(PyPolarsErr::from)?;
                let categories = ca.downcast_iter().next().unwrap().clone();
                create_enum_data_type(categories)
            },
            "Date" => DataType::Date,
            "Time" => DataType::Time,
            "Datetime" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<PyTimeUnit>()?.0;
                let time_zone = ob.getattr(intern!(py, "time_zone")).unwrap();
                let time_zone = time_zone.extract()?;
                DataType::Datetime(time_unit, time_zone)
            },
            "Duration" => {
                let time_unit = ob.getattr(intern!(py, "time_unit")).unwrap();
                let time_unit = time_unit.extract::<PyTimeUnit>()?.0;
                DataType::Duration(time_unit)
            },
            #[cfg(feature = "dtype-decimal")]
            "Decimal" => {
                let precision = ob.getattr(intern!(py, "precision"))?.extract()?;
                let scale = ob.getattr(intern!(py, "scale"))?.extract()?;
                DataType::Decimal(precision, Some(scale))
            },
            "List" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let inner = inner.extract::<PyDataType>()?;
                DataType::List(Box::new(inner.0))
            },
            #[cfg(feature = "dtype-array")]
            "Array" => {
                let inner = ob.getattr(intern!(py, "inner")).unwrap();
                let size = ob.getattr(intern!(py, "size")).unwrap();
                let inner = inner.extract::<PyDataType>()?;
                let size = size.extract::<usize>()?;
                DataType::Array(Box::new(inner.0), size)
            },
            #[cfg(feature = "dtype-struct")]
            "Struct" => {
                let fields = ob.getattr(intern!(py, "fields"))?;
                let fields = fields
                    .extract::<Vec<PyField>>()?
                    .into_iter()
                    .map(|f| f.0)
                    .collect::<Vec<Field>>();
                DataType::Struct(fields)
            },
            "Null" => DataType::Null,
            #[cfg(feature = "object")]
            "Object" => panic!("object not supported"),
            "Unknown" => DataType::Unknown(Default::default()),
            dt => {
                return Err(PyTypeError::new_err(format!(
                    "'{dt}' is not a Polars data type, or the plugin isn't compiled with the right features",
                )))
            },
        };
        Ok(PyDataType(dtype))
    }
}
