import pytest
from func_85 import add  # replace 'your_module' with the name of the module where the function is defined
def test_add():
    assert add([4, 2, 6, 7]) == 2
    assert add([1, 2, 3, 4, 5]) == 4
    assert add([2, 2, 2, 2]) == 8
    assert add([1, 1, 1, 1]) == 0
    assert add([0, 1, 2, 3, 4]) == 2

