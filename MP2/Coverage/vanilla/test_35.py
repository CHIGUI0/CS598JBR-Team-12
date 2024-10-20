import pytest
from func_35 import max_element  # replace 'your_module' with the actual module name
def test_max_element():
    assert max_element([1, 2, 3]) == 3
    assert max_element([5, 3, -5, 2, -3, 3, 9, 0, 123, 1, -10]) == 123
    assert max_element([0]) == 0
    assert max_element([-1, -2, -3]) == -1
    assert max_element([1, 1, 1, 1]) == 1

