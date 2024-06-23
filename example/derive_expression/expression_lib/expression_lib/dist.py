import polars as pl
from polars.type_aliases import IntoExpr
from polars.plugins import register_plugin_function
from pathlib import Path


from expression_lib.utils import parse_into_expr

def hamming_distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr, other],
        function_name="hamming_distance",
        is_elementwise=True,
    )


def jaccard_similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[expr, other],
        function_name="jaccard_similarity",
        is_elementwise=True,
    )


def haversine(
    start_lat: IntoExpr,
    start_long: IntoExpr,
    end_lat: IntoExpr,
    end_long: IntoExpr,
) -> pl.Expr:
    start_lat = parse_into_expr(start_lat)
    return register_plugin_function(
        plugin_path=Path(__file__).parent,
        args=[start_lat, start_long, end_lat, end_long],
        function_name="haversine",
        is_elementwise=True,
        cast_to_supertype=True,
    )
