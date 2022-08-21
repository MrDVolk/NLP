import warnings


def ignore_warnings(func, warning_types=None):
    with warnings.catch_warnings():
        if warning_types is None:
            warnings.simplefilter('ignore')
        else:
            for warning_type in warning_types:
                warnings.simplefilter("ignore", warning_type)
        return func()
