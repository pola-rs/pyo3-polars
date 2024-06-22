import polars as pl
from polars.type_aliases import IntoExpr
from polars.plugins import register_plugin_function

from expression_lib.utils import parse_into_expr
from pathlib import Path


def panic(expr: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr],
        function_name="panic",
    )
