import pytest
from func_52 import below_threshold
def test_empty_list():
    assert below_threshold([], 100) == True

def test_all_numbers_below_threshold():
    assert below_threshold([1, 2, 3, 4, 5], 6) == True

def test_some_numbers_above_threshold():
    assert below_threshold([1, 2, 6, 4, 5], 5) == False

def test_negative_numbers():
    assert below_threshold([-1, -2, -3, -4, -5], 0) == True

def test_zero():
    assert below_threshold([0, 1, 2, 3, 4, 5], 1) == False

def test_boundary_checks():
    assert below_threshold([1, 2, 3, 4, 5], 1) == False
    assert below_threshold([1, 2, 3, 4, 5], 5) == True
