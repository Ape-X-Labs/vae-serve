from loguru import logger


def run_if(condition_func):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if condition_func(*args, **kwargs):
                return func(*args, **kwargs)
            else:
                logger.warning("Condition not met. Function not executed.")
        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        return wrapper

    return decorator
