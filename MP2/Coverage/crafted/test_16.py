import pytest
from func_16 import count_distinct_characters
def test_empty_string():
    assert count_distinct_characters("") == 0

def test_single_character_repeated():
    assert count_distinct_characters("aaaaa") == 1

def test_multiple_characters_different_cases():
    assert count_distinct_characters("aBcABC") == 3

def test_special_characters():
    assert count_distinct_characters("!@#$%^&*()") == 10

def test_numbers():
    assert count_distinct_characters("1234567890") == 10

def test_boundary_checks():
    assert count_distinct_characters("a") == 1
    assert count_distinct_characters("aa") == 1
    assert count_distinct_characters("aaa") == 1

def test_various_lengths_and_characters():
    assert count_distinct_characters("abc") == 3
    assert count_distinct_characters("aabbcc") == 3
    assert count_distinct_characters("123456") == 6
    assert count_distinct_characters("111222333") == 3

