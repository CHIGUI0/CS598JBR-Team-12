import pytest
from func_51 import remove_vowels
def test_empty_string():
    assert remove_vowels("") == ""

def test_string_with_vowels():
    assert remove_vowels("aeiou") == ""

def test_string_without_vowels():
    assert remove_vowels("bcdf") == "bcdf"

def test_string_with_special_characters():
    assert remove_vowels("!@#$%^&*()") == "!@#$%^&*()"

def test_case_sensitivity():
    assert remove_vowels("AEIOU") == ""

def test_boundary_checks():
    assert remove_vowels("a") == ""
    assert remove_vowels("bcdf") == "bcdf"

def test_various_lengths_and_characters():
    assert remove_vowels("abcdefghijklmnopqrstuvwxyz") == "bcdfghjklmnpqrstvwxyz"
    assert remove_vowels("AEIOU") == ""

