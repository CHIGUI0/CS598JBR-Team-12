import pytest
from func_52 import below_threshold  # replace 'your_module' with the actual module name
def test_below_threshold():
    assert below_threshold([1, 2, 4, 10], 100)
    assert not below_threshold([1, 20, 4, 10], 5)
    assert not below_threshold([1, 20, 4, 10], 20)
    assert below_threshold([1, 2, 3, 4], 5)
    assert not below_threshold([1, 2, 3, 4], 4)

