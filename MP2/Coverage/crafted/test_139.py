import pytest
from func_139 import special_factorial
def test_zero():
    assert special_factorial(0) == 1

def test_positive_integers():
    assert special_factorial(5) == 3400

def test_negative_integers():
    assert special_factorial(-5) == 1

def test_special_cases():
    assert special_factorial(1000) == 1  # This test will take a long time to run

def test_boundary_checks():
    assert special_factorial(1) == 1
    assert special_factorial(2) == 4
    assert special_factorial(3) == 12
    assert special_factorial(4) == 288

