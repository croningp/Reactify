"""Helper module to expose function parameters as command-line arguments.

Uses type annotations and default values for keyword arguments to parse
user-provided command-line arguments to the desired type.

```python
import argparse

from Reactify.util import register_params, retrieve_params


def read_input(
    in_dir: str,
    recursive=False,
    num_workers=None:int, # type annotations take precedence over default values
):
    pass

def main(out_file: str, io_params={}, data_params={}):
    data = read_input(**io_params)
    results = process_data(data, **data_params)
    save(results, out_file)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    register_params(parser, read_input)
    register_params(parser, process_data)
    register_params(parser, main, exclude=["io_params", "data_params"])
    args = parser.parse_args()
    main(
        **retrieve_params(args, main),
        io_params=retrieve_params(args, read_input)
        data_params=retrieve_params(args, process_data)
    )
```
"""

import argparse
import inspect
from typing import Callable, Iterable


def register_params(
    parser: argparse.ArgumentParser, f: Callable, exclude: Iterable = (),
):
    params = inspect.signature(f).parameters
    for name, param in params.items():
        kwargs = {}
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            kwargs.update(nargs="*")
        if name in exclude:
            continue
        default = param.default
        param_type = (
            param.annotation if param.annotation != param.empty else type(default)
        )
        if param_type == bool:
            if default is True:
                parser.add_argument(
                    f"--no-{name}", dest=name, action="store_false", **kwargs
                )
            else:
                parser.add_argument(
                    f"--{name}", dest=name, action="store_true", **kwargs
                )
        elif default != param.empty:
            parser.add_argument(
                f"--{name}",
                type=param_type,
                default=default,
                help=f"default: {default} type:{param_type.__name__}",
                **kwargs,
            )
        else:
            parser.add_argument(
                f"{name}", type=param_type, help=f"type:{param_type.__name__}", **kwargs
            )


def retrieve_args(args, f: Callable):
    result = []
    params = inspect.signature(f).parameters

    for k, w in params.items():
        if k not in args or w.kind == inspect.Parameter.KEYWORD_ONLY:
            continue
        if w.kind == inspect.Parameter.VAR_POSITIONAL:
            result.extend(args.__getattribute__(k))
        else:
            result.append(args.__getattribute__(k))

    return result


def retrieve_kwargs(args, f: Callable, strict=False):
    params = inspect.signature(f).parameters
    param_types = [inspect.Parameter.KEYWORD_ONLY]
    if not strict:
        param_types += [inspect.Parameter.POSITIONAL_OR_KEYWORD]
    return {
        k: args.__getattribute__(k)
        for k, w in params.items()
        if k in args and w.kind in param_types
    }
