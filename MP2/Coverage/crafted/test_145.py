import pytest
from func_145 import order_by_points
def test_empty_list():
    assert order_by_points([]) == []

def test_single_number():
    assert order_by_points([1]) == [1]

def test_multiple_numbers_same_digits_sum():
    assert order_by_points([1, 11, 111]) == [1, 11, 111]

def test_negative_numbers():
    assert order_by_points([-1, -11, -111]) == [-1, -11, -111]

def test_special_cases():
    assert order_by_points([10, 20, 30]) == [10, 20, 30]

def test_boundary_checks():
    assert order_by_points([1, 9, 10, 11]) == [1, 9, 10, 11]

def test_various_lengths_and_numbers():
    assert order_by_points([1, 11, 111, 2, 22, 222]) == [1, 2, 11, 22, 111, 222]
