import pytest
from func_144 import simplify
def test_simple_fraction():
    assert simplify("1/5", "5/1") == True

def test_whole_numbers():
    assert simplify("100/1", "100/1") == True

def test_zero_denominator():
    assert simplify("1/1", "0/1") == False

def test_fractions_different_denominators():
    assert simplify("1/2", "2/4") == True

def test_fractions_different_numerators():
    assert simplify("1/2", "1/3") == False

def test_fractions_different_numerators_and_denominators():
    assert simplify("1/2", "3/4") == False

def test_boundary_checks():
    assert simplify("1/2", "2/1") == False
    assert simplify("100/1", "100/1") == True

