import polars as pl
from polars.type_aliases import IntoExpr
from polars.plugins import register_plugin_function
from pathlib import Path

from expression_lib.utils import parse_into_expr


def is_leap_year(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr],
        function_name="is_leap_year",
        is_elementwise=True,
    )


# Note that this already exists in Polars. It is just for explanatory
# purposes.
def change_time_zone(expr: IntoExpr, tz: str = "Europe/Amsterdam") -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr],
        function_name="change_time_zone", is_elementwise=True, kwargs={"tz": tz}
    )
