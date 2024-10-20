import pytest
from func_57 import monotonic  # replace 'your_module' with the actual module name
def test_monotonic_increasing():
    assert monotonic([1, 2, 4, 20])

def test_monotonic_decreasing():
    assert monotonic([4, 1, 0, -10])

def test_monotonic_false():
    assert not monotonic([1, 20, 4, 10])

