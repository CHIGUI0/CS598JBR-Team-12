import pytest
from func_85 import add
def test_empty_list():
    assert add([]) == 0

def test_single_element():
    assert add([4]) == 0

def test_multiple_elements_with_even_numbers_at_odd_indices():
    assert add([1, 2, 3, 4, 5]) == 2

def test_multiple_elements_with_at_least_one_even_number_at_odd_index():
    assert add([1, 2, 3, 4, 5]) == 2

def test_special_numbers():
    assert add([1.5, 2.5, 3.5, 4.5, 5.5]) == 0

def test_negative_numbers():
    assert add([-1, -2, -3, -4, -5]) == 0

def test_various_lengths_and_numbers():
    assert add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 2

def test_boundary_checks():
    assert add([1]) == 0
    assert add([1, 2]) == 2
    assert add([1, 2, 3]) == 2
    assert add([1, 2, 3, 4]) == 2
    assert add([1, 2, 3, 4, 5]) == 2
    assert add([1, 2, 3, 4, 5, 6]) == 2
    assert add([1, 2, 3, 4, 5, 6, 7]) == 2
    assert add([1, 2, 3, 4, 5, 6, 7, 8]) == 2
    assert add([1, 2, 3, 4, 5, 6, 7, 8, 9]) == 2
    assert add([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) == 2
