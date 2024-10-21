import pytest
from func_78 import hex_key
def test_empty_string():
    assert hex_key("") == 0

def test_single_prime():
    assert hex_key("2") == 1

def test_multiple_characters_with_prime_numbers():
    assert hex_key("ABED1A33") == 4

def test_multiple_characters_with_at_least_one_non_prime_number():
    assert hex_key("123456789ABCDEF0") == 6

def test_special_characters():
    assert hex_key("ABED1A33") == 4

def test_case_sensitivity():
    assert hex_key("abed1a33") == 4

def test_boundary_checks():
    assert hex_key("2020") == 2
