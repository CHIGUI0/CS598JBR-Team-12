import pytest
from func_57 import monotonic
def test_empty_list():
    assert monotonic([]) == True

def test_single_element_list():
    assert monotonic([1]) == True

def test_increasing_sequence():
    assert monotonic([1, 2, 3, 4]) == True

def test_decreasing_sequence():
    assert monotonic([4, 3, 2, 1]) == True

def test_mixed_sequence():
    assert monotonic([1, 2, 3, 2, 1]) == True

def test_negative_numbers():
    assert monotonic([1, -2, -3, -2, 1]) == True

def test_zero():
    assert monotonic([1, 0, -1, -2, -3]) == True

def test_non_monotonic_sequence():
    assert monotonic([1, 2, 3, 2, 1]) == False

def test_non_monotonic_negative_numbers():
    assert monotonic([1, -2, -3, -2, 1]) == False

def test_non_monotonic_zero():
    assert monotonic([1, 0, -1, -2, -3]) == False
