import pytest
from func_35 import max_element
def test_empty_list():
    assert max_element([]) == None

def test_single_element_list():
    assert max_element([5]) == 5

def test_list_with_negative_numbers():
    assert max_element([-5, -3, -1]) == -1

def test_list_with_repeated_maximum():
    assert max_element([5, 3, 5, 1]) == 5

def test_list_with_special_numbers():
    assert max_element([5, float('inf'), 3]) == float('inf')

def test_boundary_check_smallest_non_empty_list():
    assert max_element([1]) == 1

def test_boundary_check_largest_non_empty_list():
    assert max_element([1, 2, 3, 4, 5]) == 5

def test_boundary_check_maximum_transition_from_one_element_to_another():
    assert max_element([1, 2, 3, 4, 5]) == 5
