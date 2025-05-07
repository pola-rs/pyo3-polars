import polars as pl

from extend_polars import (
    parallel_jaccard,
    lazy_parallel_jaccard,
    debug,
    expr_debug,
    lazy_debug,
)

df = pl.DataFrame(
    {"list_a": [[1, 2, 3], [5, 5]], "list_b": [[1, 2, 3, 8], [5, 1, 1]]}
)

print(df)
print(parallel_jaccard(df, "list_a", "list_b"))

# warning this serializes/deserialized the data
# it is recommended to only use LazyFrames that don't have any
# DataFrame in their logical plan.
print(lazy_debug(df.lazy()))
print(lazy_parallel_jaccard(df.lazy(), "list_a", "list_b").collect())

df = pl.DataFrame({"string": ["ab", "c"]})
print(debug(df))

expr = pl.col("some_col").filter(pl.col("some_col").ge(pl.lit(1_000)))
print(expr_debug(expr))
