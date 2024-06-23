import polars as pl
from polars.type_aliases import IntoExpr
from polars.plugins import register_plugin_function
from pathlib import Path

from expression_lib.utils import parse_into_expr



def pig_latinnify(expr: IntoExpr, capitalize: bool = False) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr],
        function_name="pig_latinnify",
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
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr],
        kwargs={
            "float_arg": float_arg,
            "integer_arg": integer_arg,
            "string_arg": string_arg,
            "boolean_arg": boolean_arg,
        },
        function_name="append_kwargs",
        is_elementwise=True,
    )
