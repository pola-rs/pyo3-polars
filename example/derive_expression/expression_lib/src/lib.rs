use std::path::PathBuf;
use pyo3::prelude::*;

mod expressions;

// #[pyfunction]
// fn lib_exe() -> PyResult<PathBuf> {
//     std::env::current_dir()
//     let p = std::env::current_exe()?;
//     Ok(p)
// }
//
// #[pymodule]
// fn expression_lib(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_function(wrap_pyfunction!(lib_exe, m)?)?;
//     Ok(())
// }
