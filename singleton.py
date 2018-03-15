
"""
from:
https://stackoverflow.com/questions/6760685/creating-a-singleton-in-python

use like this:

#Python2
class MyClass(BaseClass):
    __metaclass__ = Singleton

#Python3
class MyClass(BaseClass, metaclass=Singleton):
    pass
"""


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]
