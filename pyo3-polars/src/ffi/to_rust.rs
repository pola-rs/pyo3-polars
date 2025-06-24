use crate::error::PyPolarsErr;
use crate::PySeries;
use polars::prelude::*;
use polars_arrow::array::Array;
use polars_arrow::ffi;
use pyo3::exceptions::PyValueError;
use pyo3::ffi::Py_uintptr_t;
use pyo3::prelude::*;
use pyo3::types::PyCapsule;

pub fn array_to_rust(obj: &Bound<PyAny>) -> PyResult<ArrayRef> {
    // prepare a pointer to receive the Array struct
    let array = Box::new(ffi::ArrowArray::empty());
    let schema = Box::new(ffi::ArrowSchema::empty());

    let array_ptr = &*array as *const ffi::ArrowArray;
    let schema_ptr = &*schema as *const ffi::ArrowSchema;

    // make the conversion through PyArrow's private API
    // this changes the pointer's memory and is thus unsafe. In particular, `_export_to_c` can go out of bounds
    obj.call_method1(
        "_export_to_c",
        (array_ptr as Py_uintptr_t, schema_ptr as Py_uintptr_t),
    )?;

    unsafe {
        let field = ffi::import_field_from_c(schema.as_ref()).map_err(PyPolarsErr::from)?;
        let array = ffi::import_array_from_c(*array, field.dtype).map_err(PyPolarsErr::from)?;
        Ok(array)
    }
}

/// Import `__arrow_c_stream__` across Python boundary.
pub fn call_arrow_c_stream<'py>(ob: &'py Bound<PyAny>) -> PyResult<Bound<'py, PyCapsule>> {
    if !ob.hasattr("__arrow_c_stream__")? {
        return Err(PyValueError::new_err(
            "Expected an object with dunder __arrow_c_stream__",
        ));
    }

    let capsule = ob.getattr("__arrow_c_stream__")?.call0()?.downcast_into()?;
    Ok(capsule)
}

/// Validate PyCapsule has provided name
pub fn validate_pycapsule_name(capsule: &Bound<PyCapsule>, expected_name: &str) -> PyResult<()> {
    let capsule_name = capsule.name()?;
    if let Some(capsule_name) = capsule_name {
        let capsule_name = capsule_name.to_str()?;
        if capsule_name != expected_name {
            return Err(PyValueError::new_err(format!(
                "Expected name '{}' in PyCapsule, instead got '{}'",
                expected_name, capsule_name
            )));
        }
    } else {
        return Err(PyValueError::new_err(
            "Expected schema PyCapsule to have name set.",
        ));
    }

    Ok(())
}

pub fn import_stream_pycapsule(capsule: &Bound<PyCapsule>) -> PyResult<PySeries> {
    validate_pycapsule_name(capsule, "arrow_array_stream")?;
    // # Safety
    // capsule holds a valid C ArrowArrayStream pointer, as defined by the Arrow PyCapsule
    // Interface
    let mut stream = unsafe {
        // Takes ownership of the pointed to ArrowArrayStream
        // This acts to move the data out of the capsule pointer, setting the release callback to NULL
        let stream_ptr = Box::new(std::ptr::replace(
            capsule.pointer() as _,
            ffi::ArrowArrayStream::empty(),
        ));
        ffi::ArrowArrayStreamReader::try_new(stream_ptr)
            .map_err(|err| PyValueError::new_err(err.to_string()))?
    };

    let mut produced_arrays: Vec<Box<dyn Array>> = vec![];
    while let Some(array) = unsafe { stream.next() } {
        produced_arrays.push(array.unwrap());
    }

    // Series::try_from fails for an empty vec of chunks
    let s = if produced_arrays.is_empty() {
        let polars_dt = DataType::from_arrow_field(stream.field());
        Series::new_empty(stream.field().name.clone(), &polars_dt)
    } else {
        Series::try_from((stream.field(), produced_arrays)).unwrap()
    };
    Ok(PySeries(s))
}
