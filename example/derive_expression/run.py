import polars as pl
from expression_lib import Language

df = pl.DataFrame({
    "names": ["Richard", "Alice", "Bob"]
})

print(df.with_columns(
   pl.col("names").language.pig_latinnify()
))
