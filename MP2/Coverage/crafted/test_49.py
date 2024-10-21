import pytest
from func_49 import modp
def test_small_n():
    assert modp(3, 101) == 3

def test_large_n():
    assert modp(1000, 101) == 1

def test_p_power_of_2():
    assert modp(3, 8) == 0

def test_p_not_power_of_2():
    assert modp(3, 11) == 8

def test_n_is_0():
    assert modp(0, 101) == 1

def test_various_n_and_p():
    assert modp(5, 13) == 12
    assert modp(7, 19) == 16

def test_boundary_checks():
    assert modp(0, 1) == 0
    assert modp(1000, 1) == 0
    assert modp(1, 1000) == 1
    assert modp(1, 1) == 0
