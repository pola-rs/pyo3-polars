import polars as pl
from polars.type_aliases import IntoExpr
from polars.utils.udfs import _get_shared_lib_location

lib = _get_shared_lib_location(__file__)


@pl.api.register_expr_namespace("language")
class Language:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def pig_latinnify(self) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            symbol="pig_latinnify",
            is_elementwise=True,
        )

    def append_args(
        self,
        float_arg: float,
        integer_arg: int,
        string_arg: str,
        boolean_arg: bool,
        dict_arg: dict,
    ) -> pl.Expr:
        """
        This example shows how arguments other than `Series` can be used.
        """
        return self._expr._register_plugin(
            lib=lib,
            args=[],
            kwargs={
                "float_arg": float_arg,
                "integer_arg": integer_arg,
                "string_arg": string_arg,
                "boolean_arg": boolean_arg,
                "dict_arg": dict_arg,
            },
            symbol="append_kwargs",
            is_elementwise=True,
        )


@pl.api.register_expr_namespace("dist")
class Distance:
    def __init__(self, expr: pl.Expr):
        self._expr = expr

    def hamming_distance(self, other: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[other],
            symbol="hamming_distance",
            is_elementwise=True,
        )

    def jaccard_similarity(self, other: IntoExpr) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[other],
            symbol="jaccard_similarity",
            is_elementwise=True,
        )

    def haversine(
        self,
        start_lat: IntoExpr,
        start_long: IntoExpr,
        end_lat: IntoExpr,
        end_long: IntoExpr,
    ) -> pl.Expr:
        return self._expr._register_plugin(
            lib=lib,
            args=[start_lat, start_long, end_lat, end_long],
            symbol="haversine",
            is_elementwise=True,
            cast_to_supertypes=True,
        )
