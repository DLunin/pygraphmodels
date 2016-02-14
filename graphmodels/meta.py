from singledispatch import singledispatch, update_wrapper
from functools import wraps
from copy import deepcopy


def methoddispatch(func):
    # http://stackoverflow.com/questions/24601722/how-can-i-use-functools-singledispatch-with-instance-methods
    dispatcher = singledispatch(func)

    def wrapper(*args, **kwargs):
        return dispatcher.dispatch(args[1].__class__)(*args, **kwargs)

    wrapper.register = dispatcher.register
    update_wrapper(wrapper, func)
    return wrapper


def copy_option(default=False):
    def copy_option_decorator(f):
        @wraps(f)
        def new_f(self, *args, **kwargs):
            if 'copy' in kwargs and kwargs['copy']:
                del kwargs['copy']
                return f(self.copy(), *args, **kwargs)
            if 'copy' in kwargs:
                del kwargs['copy']
                return f(self, *args, **kwargs)
            return f(deepcopy(self) if default else self, *args, **kwargs)

        return new_f

    return copy_option_decorator

