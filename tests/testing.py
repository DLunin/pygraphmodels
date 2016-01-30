from functools import wraps


class AssumptionFailed(Exception):
    pass


def assume(condition):
    if not condition:
        raise AssumptionFailed()


class StochasticTestFailed(Exception):
    def __init__(self, args, kwargs):
        Exception.__init__(self, 'counterexample args=%s, kwargs=%s' % (str(args), str(kwargs)))
        self.args = args
        self.kwargs = kwargs


def stochastic_test(*arg_distr, **kwarg_distr):
    def stochastic_test_decorator(f):
        @wraps(f)
        def new_f(__test_n__, **additional_params):
            n = __test_n__
            i = 0
            while i < n:
                kwargs = {name: distr.rvs() for name, distr in kwarg_distr.items()}
                args = tuple(distr.rvs() for distr in arg_distr)
                kwargs.update(additional_params)
                try:
                    f(*args, **kwargs)
                except AssumptionFailed:
                    continue
                except:
                    raise StochasticTestFailed(args, kwargs)
                i += 1
            return None

        new_f.tested_function = f
        return new_f

    return stochastic_test_decorator


def convergence_test(**kwargs):
    assert len(kwargs) == 1
    var, values = tuple(kwargs.items())[0]

    def convergence_test_decorator(test):
        @wraps(test)
        def new_test(__test_n__, **additional_kwargs):
            n = __test_n__
            counterexamples = []
            for value in values:
                test_kwargs = additional_kwargs.copy()
                test_kwargs.update({var: value})
                try:
                    test(n, **test_kwargs)
                except StochasticTestFailed as ex:
                    counterexamples.append(ex)
                    continue
                failed = False
                for exception in counterexamples:
                    try:
                        test_kwargs = exception.kwargs
                        test_kwargs.update({var: value})
                        test.tested_function(*exception.args, **exception.kwargs)
                    except:
                        failed = True
                        break
                if not failed:
                    return None
            raise counterexamples[-1]

        return new_test

    return convergence_test_decorator
