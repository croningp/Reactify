import argparse
import inspect
from typing import Callable, Iterable

def register_params(parser: argparse.ArgumentParser, f: Callable, exclude: Iterable=()):
    params = inspect.signature(f).parameters
    for name, param in params.items():
        if name in exclude:
            continue
        default = param.default
        param_type = (
            param.annotation if param.annotation != param.empty else type(default)
        )
        if default != param.empty:
            parser.add_argument(
                f"--{name}",
                type=param_type,
                default=default,
                help=f"default: {default} type:{param_type.__name__}",
            )
        else:
            parser.add_argument(
                f"{name}", type=param_type, help=f"type:{param_type.__name__}"
            )

def retrieve_params(args, f: Callable):
    params = inspect.signature(f).parameters
    return {param: args.__getattribute__(param) for param in params if param in args}