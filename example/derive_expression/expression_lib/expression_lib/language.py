import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

from expression_lib.utils import parse_into_expr

lib = _get_shared_lib_location(__file__)


def pig_latinnify(expr: IntoExpr, capitalize: bool = False) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        symbol="pig_latinnify",
        is_elementwise=True,
        kwargs={"capitalize": capitalize},
    )


def append_args(
    expr: IntoExpr,
    float_arg: float,
    integer_arg: int,
    string_arg: str,
    boolean_arg: bool,
) -> pl.Expr:
    """
    This example shows how arguments other than `Series` can be used.
    """
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        args=[],
        kwargs={
            "float_arg": float_arg,
            "integer_arg": integer_arg,
            "string_arg": string_arg,
            "boolean_arg": boolean_arg,
        },
        symbol="append_kwargs",
        is_elementwise=True,
    )
