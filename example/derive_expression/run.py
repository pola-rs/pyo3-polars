import polars as pl
from expression_lib import Language, Distance
from datetime import date

df = pl.DataFrame(
    {
        "names": ["Richard", "Alice", "Bob"],
        "moons": ["full", "half", "red"],
        "dates": [date(2023, 1, 1), date(2024, 1, 1), date(2025, 1, 1)],
        "dist_a": [[12, 32, 1], [], [1, -2]],
        "dist_b": [[-12, 1], [43], [876, -45, 9]],
        "floats": [5.6, -1245.8, 242.224],
    }
)


out = df.with_columns(
    pig_latin=pl.col("names").language.pig_latinnify(),
    pig_latin_cap=pl.col("names").language.pig_latinnify(capitalize=True),
).with_columns(
    hamming_dist=pl.col("names").dist.hamming_distance("pig_latin"),
    jaccard_sim=pl.col("dist_a").dist.jaccard_similarity("dist_b"),
    haversine=pl.col("floats").dist.haversine("floats", "floats", "floats", "floats"),
    leap_year=pl.col("dates").date_util.is_leap_year(),
    appended_args=pl.col("names").language.append_args(
        float_arg=11.234,
        integer_arg=93,
        boolean_arg=False,
        string_arg="example",
    ),
)

print(out)


# Tests we can return errors from FFI by passing wrong types.
try:
    out.with_columns(
        appended_args=pl.col("names").language.append_args(
            float_arg=True,
            integer_arg=True,
            boolean_arg=True,
            string_arg="example",
        )
    )
except pl.ComputeError as e:
    assert "the plugin failed with message" in str(e)
