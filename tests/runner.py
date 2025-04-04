import os 
os.environ['LOGURU_LEVEL'] = 'DEBUG'
from typing import List, Callable
from loguru import logger
from traceback import format_exc
import os
import importlib
import inspect


def runner(functions: List[Callable]):
    logger.info("Running tests...")
    for function in functions:
        try:
            logger.info(f"Running test {function.__name__}...")
            function()
        except Exception as e:
            logger.error(f"❌ Test {function.__name__} failed.")
            logger.error(format_exc())
        else:
            logger.info(f"✅ Test {function.__name__} passed.")
    logger.info("Tests completed.")

if __name__ == "__main__":
    # look for and load all files that start with test_ and end with _test.py
    tests = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    for file in os.listdir(current_dir):
        if file.startswith("test_") or file.endswith("_test.py"):
            # remove the tests/ prefix
            path = os.path.join(current_dir, file)
            module_name = path.replace("/", ".").replace(".py", "")
            module_name = module_name.split(".")[-2:]
            module_name = ".".join(module_name)
            logger.info(f"Loading test file: {module_name}")
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module):
                if inspect.isfunction(obj) and name.startswith("test_"):
                    tests.append(obj)
    runner(tests)
