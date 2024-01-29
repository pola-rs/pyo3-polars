import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

from expression_lib.utils import parse_into_expr

lib = _get_shared_lib_location(__file__)


def is_leap_year(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="is_leap_year",
        is_elementwise=True,
    )


# Note that this already exists in Polars. It is just for explanatory
# purposes.
def change_time_zone(expr: IntoExpr, tz: str = "Europe/Amsterdam") -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib, symbol="change_time_zone", is_elementwise=True, kwargs={"tz": tz}
    )
