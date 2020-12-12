"""Helper module to expose function parameters as command-line arguments.

Uses type annotations and default values for keyword arguments to parse
user-provided command-line arguments to the desired type.

```python
import argparse

from reactify.util import register_params, retrieve_params


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
    parser: argparse.ArgumentParser, f: Callable, exclude: Iterable = ()
):
    params = inspect.signature(f).parameters
    for name, param in params.items():
        if name in exclude:
            continue
        default = param.default
        param_type = (
            param.annotation if param.annotation != param.empty else type(default)
        )
        if param_type == bool:
            if default == True:
                parser.add_argument(f"--no-{name}", dest=name, action="store_false")
            else:
                parser.add_argument(f"--{name}", dest=name, action="store_true")
        elif default != param.empty:
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
