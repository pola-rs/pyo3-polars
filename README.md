## 1. Shared library plugins for Polars
This is new functionality and not entirely stable, but should be preferred over `2.` as this
will circumvent the GIL and will be the way we want to support extending polars.

See more in `examples/derive_expression`.

## 2. Pyo3 extensions for Polars
  <a href="https://crates.io/crates/pyo3-polars">
    <img src="https://img.shields.io/crates/v/pyo3-polars.svg"/>
  </a>

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