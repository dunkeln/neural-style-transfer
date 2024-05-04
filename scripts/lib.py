from functools import reduce


def compose(*fns):
    def __inner__(*args, **kwargs):
        return reduce(lambda x, f: f(x), reversed(fns), *args, **kwargs)
    return __inner__
