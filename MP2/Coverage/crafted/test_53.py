import pytest
from func_53 import add
def test_add_positive_integers():
    assert add(2, 3) == 5

def test_add_negative_integers():
    assert add(-2, -3) == -5

def test_add_positive_and_negative_integers():
    assert add(-2, 3) == 1

def test_add_zero():
    assert add(0, 0) == 0
    assert add(2, 0) == 2
    assert add(0, 3) == 3

def test_add_special_cases():
    with pytest.raises(TypeError):
        add(float('inf'), float('inf'))
    with pytest.raises(TypeError):
        add(float('inf'), 1)
    with pytest.raises(TypeError):
        add(1, float('inf'))
    with pytest.raises(TypeError):
        add(float('inf'), float('nan'))
    with pytest.raises(TypeError):
        add(float('nan'), float('nan'))
    with pytest.raises(TypeError):
        add(float('nan'), 1)
    with pytest.raises(TypeError):
        add(1, float('nan'))

def test_add_various_lengths_and_numbers():
    assert add(1234567890, 9876543210) == 11111111100
    assert add(-1234567890, 9876543210) == 8641975320
    assert add(-1234567890, -9876543210) == -11111111100

def test_add_boundary_checks():
    assert add(2**63-1, 0) == 2**63-1
    assert add(-2**63, 0) == -2**63
    assert add(0, 2**63-1) == 2**63-1
    assert add(0, -2**63) == -2**63
