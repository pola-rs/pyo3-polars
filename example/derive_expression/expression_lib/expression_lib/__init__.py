import polars as pl
import os

def is_shared_lib(file: str) -> bool:
    return file.endswith(".so") or file.endswith(".dll")

directory = os.path.dirname(__file__)
lib = os.path.join(directory, next(filter(is_shared_lib, os.listdir(directory))))


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
