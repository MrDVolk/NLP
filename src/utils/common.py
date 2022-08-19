import warnings


def ignore_warnings(func, warning_types):
    with warnings.catch_warnings():
        for warning_type in warning_types:
            warnings.simplefilter("ignore", warning_type)
        return func()
