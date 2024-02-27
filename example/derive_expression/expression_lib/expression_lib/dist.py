import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

from expression_lib.utils import parse_into_expr

lib = _get_shared_lib_location(__file__)


def hamming_distance(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="hamming_distance",
        is_elementwise=True,
    )


def jaccard_similarity(expr: IntoExpr, other: IntoExpr) -> pl.Expr:
    expr = parse_into_expr(expr)
    return expr.register_plugin(
        lib=lib,
        args=[other],
        symbol="jaccard_similarity",
        is_elementwise=True,
    )


def haversine(
    start_lat: IntoExpr,
    start_long: IntoExpr,
    end_lat: IntoExpr,
    end_long: IntoExpr,
) -> pl.Expr:
    start_lat = parse_into_expr(start_lat)
    return start_lat.register_plugin(
        lib=lib,
        args=[start_long, end_lat, end_long],
        symbol="haversine",
        is_elementwise=True,
        cast_to_supertypes=True,
    )
