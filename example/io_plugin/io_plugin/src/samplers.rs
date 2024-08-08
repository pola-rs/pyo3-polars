use polars::export::arrow::bitmap::MutableBitmap;
use polars::export::arrow::types::NativeType;
use polars::prelude::*;
use pyo3::{pyclass, pyfunction};
use pyo3_polars::export::polars_core::datatypes::{DataType, PolarsDataType};
use pyo3_polars::export::polars_core::export::arrow::array::BooleanArray;
use pyo3_polars::export::polars_core::prelude::Series;
use pyo3_polars::PyDataType;
use rand::distributions::uniform::SampleUniform;
use rand::distributions::{Bernoulli, Uniform};
use rand::prelude::*;
use std::sync::Mutex;

#[pyclass]
#[derive(Clone)]
pub struct PySampler(pub Arc<Mutex<Box<dyn Sampler>>>);

pub trait Sampler: Send {
    fn name(&self) -> &str;

    fn dtype(&self) -> DataType;

    fn next_n(&mut self, n: usize) -> Series;
}

struct UniformSampler<X: SampleUniform + NativeType + Send> {
    name: String,
    rng: StdRng,
    d: Uniform<X>,
}

fn new_uniform_impl<T: NumericNative + SampleUniform>(
    name: String,
    low: T,
    high: T,
    seed: u64,
) -> UniformSampler<T> {
    UniformSampler {
        name,
        rng: StdRng::seed_from_u64(seed),
        d: Uniform::new(low, high),
    }
}

impl<T: NumericNative + SampleUniform + Send> Sampler for UniformSampler<T>
where
    Series: NamedFromOwned<Vec<T>>,
    T::Sampler: Send,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DataType {
        T::PolarsType::get_dtype()
    }

    fn next_n(&mut self, n: usize) -> Series {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let v = self.d.sample(&mut self.rng);
            out.push(v);
        }
        Series::from_vec(self.name(), out)
    }
}

#[pyfunction]
pub fn new_uniform(name: String, low: f64, high: f64, dtype: PyDataType, seed: u64) -> PySampler {
    let sampler = match dtype.0 {
        DataType::Int32 => {
            let low = low as i32;
            let high = high as i32;
            Box::new(new_uniform_impl(name, low, high, seed)) as Box<dyn Sampler>
        }
        DataType::Int64 => {
            let low = low as i64;
            let high = high as i64;
            Box::new(new_uniform_impl(name, low, high, seed)) as Box<dyn Sampler>
        }
        DataType::Float64 => Box::new(new_uniform_impl(name, low, high, seed)),
        _ => todo!(),
    };
    PySampler(Arc::new(Mutex::new(sampler)))
}
struct BernoulliSample {
    name: String,
    rng: StdRng,
    d: Bernoulli,
}

impl Sampler for BernoulliSample {
    fn name(&self) -> &str {
        &self.name
    }

    fn dtype(&self) -> DataType {
        DataType::Boolean
    }

    fn next_n(&mut self, n: usize) -> Series {
        let mut bits = MutableBitmap::with_capacity(n);

        for _ in 0..n {
            let v = self.d.sample(&mut self.rng);
            bits.push(v)
        }

        Series::from_arrow(
            self.name(),
            BooleanArray::from_data_default(bits.freeze(), None).boxed(),
        )
        .unwrap()
    }
}

#[pyfunction]
pub fn new_bernoulli(name: String, p: f64, seed: u64) -> PySampler {
    let b = BernoulliSample {
        name,
        rng: StdRng::seed_from_u64(seed),
        d: Bernoulli::new(p).expect("invalid p"),
    };

    PySampler(Arc::new(Mutex::new(Box::new(b))))
}
