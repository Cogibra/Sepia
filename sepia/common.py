
def query_kwargs(key: str, default: any, **kwargs):

    if key in kwargs.keys():
        return kwargs[key]
    else:
        return default
