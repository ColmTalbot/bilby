from scipy._lib._array_api import array_namespace


def xp_wrap(func):

    def wrapped(self, *args, **kwargs):
        if "xp" not in kwargs:
            try:
                kwargs["xp"] = array_namespace(*args)
            except TypeError:
                pass
        return func(self, *args, **kwargs)

    return wrapped
