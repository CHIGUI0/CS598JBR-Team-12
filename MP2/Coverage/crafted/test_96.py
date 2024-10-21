import pytest
from func_96 import count_up_to
def test_zero():
    assert count_up_to(0) == []

def test_single_number():
    assert count_up_to(2) == [2]

def test_multiple_numbers():
    assert count_up_to(5) == [2, 3]

def test_prime_number():
    assert count_up_to(11) == [2, 3, 5, 7]

def test_non_prime_number():
    assert count_up_to(10) == [2, 3, 5, 7]

def test_boundary_check():
    assert count_up_to(1) == []

def test_boundary_check_prime():
    assert count_up_to(2) == [2]

def test_boundary_check_non_prime():
    assert count_up_to(4) == [2, 3]
