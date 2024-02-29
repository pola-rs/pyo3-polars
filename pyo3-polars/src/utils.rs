use crate::py_modules::UTILS;
use crate::PyAnyValue;
use polars::prelude::*;
use polars_core::utils::try_get_supertype;
use pyo3::intern;
use pyo3::types::PyDict;
use pyo3::{IntoPy, PyAny, PyObject, PyResult, Python};

pub(crate) fn any_values_to_dtype(column: &[AnyValue]) -> PolarsResult<(DataType, usize)> {
    // we need an index-map as the order of dtypes influences how the
    // struct fields are constructed.
    let mut types_set = PlIndexSet::new();
    for val in column.iter() {
        types_set.insert(val.into());
    }
    let n_types = types_set.len();
    Ok((types_set_to_dtype(types_set)?, n_types))
}

fn types_set_to_dtype(types_set: PlIndexSet<DataType>) -> PolarsResult<DataType> {
    types_set
        .into_iter()
        .map(Ok)
        .reduce(|a, b| try_get_supertype(&a?, &b?))
        .unwrap()
}

pub(crate) fn abs_decimal_from_digits(
    digits: impl IntoIterator<Item = u8>,
    exp: i32,
) -> Option<(i128, usize)> {
    const MAX_ABS_DEC: i128 = 10_i128.pow(38) - 1;
    let mut v = 0_i128;
    for (i, d) in digits.into_iter().map(i128::from).enumerate() {
        if i < 38 {
            v = v * 10 + d;
        } else {
            v = v.checked_mul(10).and_then(|v| v.checked_add(d))?;
        }
    }
    // we only support non-negative scale (=> non-positive exponent)
    let scale = if exp > 0 {
        // the decimal may be in a non-canonical representation, try to fix it first
        v = 10_i128
            .checked_pow(exp as u32)
            .and_then(|factor| v.checked_mul(factor))?;
        0
    } else {
        (-exp) as usize
    };
    // TODO: do we care for checking if it fits in MAX_ABS_DEC? (if we set precision to None anyway?)
    (v <= MAX_ABS_DEC).then_some((v, scale))
}

pub(crate) fn convert_date(ob: &PyAny) -> PyResult<PyAnyValue> {
    Python::with_gil(|py| {
        let date = UTILS
            .as_ref(py)
            .getattr(intern!(py, "_date_to_pl_date"))
            .unwrap()
            .call1((ob,))
            .unwrap();
        let v = date.extract::<i32>().unwrap();
        Ok(AnyValue::Date(v).into())
    })
}
pub(crate) fn convert_datetime(ob: &PyAny) -> PyResult<PyAnyValue> {
    Python::with_gil(|py| {
        // windows
        #[cfg(target_arch = "windows")]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_any_value_windows"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };
        // unix
        #[cfg(not(target_arch = "windows"))]
        let (seconds, microseconds) = {
            let convert = UTILS
                .getattr(py, intern!(py, "_datetime_for_any_value"))
                .unwrap();
            let out = convert.call1(py, (ob,)).unwrap();
            let out: (i64, i64) = out.extract(py).unwrap();
            out
        };

        // s to us
        let mut v = seconds * 1_000_000;
        v += microseconds;

        // choose "us" as that is python's default unit
        Ok(AnyValue::Datetime(v, TimeUnit::Microseconds, &None).into())
    })
}

pub(crate) fn struct_dict<'a>(
    py: Python,
    vals: impl Iterator<Item = AnyValue<'a>>,
    flds: &[Field],
) -> PyObject {
    let dict = PyDict::new(py);
    for (fld, val) in flds.iter().zip(vals) {
        dict.set_item(fld.name().as_str(), PyAnyValue(val)).unwrap()
    }
    dict.into_py(py)
}

// accept u128 array to ensure alignment is correct
pub(crate) fn decimal_to_digits(v: i128, buf: &mut [u128; 3]) -> usize {
    const ZEROS: i128 = 0x3030_3030_3030_3030_3030_3030_3030_3030;
    // safety: transmute is safe as there are 48 bytes in 3 128bit ints
    // and the minimal alignment of u8 fits u16
    let buf = unsafe { std::mem::transmute::<&mut [u128; 3], &mut [u8; 48]>(buf) };
    let mut buffer = itoa::Buffer::new();
    let value = buffer.format(v);
    let len = value.len();
    for (dst, src) in buf.iter_mut().zip(value.as_bytes().iter()) {
        *dst = *src
    }

    let ptr = buf.as_mut_ptr() as *mut i128;
    unsafe {
        // this is safe because we know that the buffer is exactly 48 bytes long
        *ptr -= ZEROS;
        *ptr.add(1) -= ZEROS;
        *ptr.add(2) -= ZEROS;
    }
    len
}
