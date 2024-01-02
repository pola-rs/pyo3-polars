## 1. Shared library plugins for Polars

<a href="https://crates.io/crates/pyo3-polars">
    <img src="https://img.shields.io/crates/v/pyo3-polars.svg"/>
  </a>

This is new functionality and should be preferred over `2.` as this
will circumvent the GIL and will be the way we want to support extending polars.

Parallelism and optimizations are managed by the default polars runtime. That runtime will call into the plugin function.
The plugin functions are compiled separately.

We can therefore keep polars more lean and maybe add support for a `polars-distance`, `polars-geo`, `polars-ml`, etc. 
Those can then have specialized expressions and don't have to worry as much for code bloat as they can be optionally installed.

The idea is that you define an expression in another Rust crate with a proc_macro `polars_expr`.

That macro can have the following attributes:

- `output_type` -> to define the output type of that expression
- `output_type_func` -> to define a function that computes the output type based on input types.

Here is an example of a `String` conversion expression that converts any string to [pig latin](https://en.wikipedia.org/wiki/Pig_Latin):

```rust
fn pig_latin_str(value: &str, capitalize: bool, output: &mut String) {
    if let Some(first_char) = value.chars().next() {
        if capitalize {
            for c in value.chars().skip(1).map(|char| char.to_uppercase()) {
                write!(output, "{c}").unwrap()
            }
            write!(output, "AY").unwrap()
        } else {
            let offset = first_char.len_utf8();
            write!(output, "{}{}ay", &value[offset..], first_char).unwrap()
        }
    }
}

#[derive(Deserialize)]
struct PigLatinKwargs {
    capitalize: bool,
}

#[polars_expr(output_type=String)]
fn pig_latinnify(inputs: &[Series], kwargs: PigLatinKwargs) -> PolarsResult<Series> {
    let ca = inputs[0].str()?;
    let out: StringChunked =
        ca.apply_to_buffer(|value, output| pig_latin_str(value, kwargs.capitalize, output));
    Ok(out.into_series())
}
```

On the python side this expression can then be registered under a namespace:

```python
import polars as pl
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("language")
class Language:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def pig_latinnify(self, capatilize: bool = False) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            symbol="pig_latinnify",
            is_elementwise=True,
            kwargs={"capitalize": capatilize}
        )
```

Compile/ship and then it is ready to use:

```python
import polars as pl
import expression_lib

df = pl.DataFrame({
    "names": ["Richard", "Alice", "Bob"],
})


out = df.with_columns(
   pig_latin = pl.col("names").language.pig_latinnify()
)
```

See the full example in [example/derive_expression]: https://github.com/pola-rs/pyo3-polars/tree/main/example/derive_expression

## 2. Pyo3 extensions for Polars

See the `example` directory for a concrete example. Here we send a polars `DataFrame` to rust and then compute a
`jaccard similarity` in parallel using `rayon` and rust hash sets.

## Run example

`$ cd example && make install`
`$ venv/bin/python run.py`

This will output:

```
shape: (2, 2)
┌───────────┬───────────────┐
│ list_a    ┆ list_b        │
│ ---       ┆ ---           │
│ list[i64] ┆ list[i64]     │
╞═══════════╪═══════════════╡
│ [1, 2, 3] ┆ [1, 2, ... 8] │
│ [5, 5]    ┆ [5, 1, 1]     │
└───────────┴───────────────┘
shape: (2, 1)
┌─────────┐
│ jaccard │
│ ---     │
│ f64     │
╞═════════╡
│ 0.75    │
│ 0.5     │
└─────────┘
```

## Compile for release

`$ make install-release`

# What to expect

This crate offers a `PySeries` and a `PyDataFrame` which are simple wrapper around `Series` and `DataFrame`. The
advantage of these wrappers is that they can be converted to and from python as they implement `FromPyObject` and `IntoPy`.
