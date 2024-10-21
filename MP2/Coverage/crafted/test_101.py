import pytest
from func_101 import words_string
def test_empty_string():
    assert words_string("") == []

def test_single_word():
    assert words_string("word") == ["word"]

def test_multiple_words_with_commas():
    assert words_string("word1,word2,word3") == ["word1", "word2", "word3"]

def test_multiple_words_with_spaces():
    assert words_string("word1 word2 word3") == ["word1", "word2", "word3"]

def test_special_characters():
    assert words_string("word1@word2#word3") == ["word1@word2#word3"]

def test_case_sensitivity():
    assert words_string("Word1 Word2 Word3") == ["Word1", "Word2", "Word3"]

def test_boundary_checks():
    assert words_string("word") == ["word"]
    assert words_string("word word word") == ["word", "word", "word"]
    assert words_string("word,word,word") == ["word", "word", "word"]
    assert words_string("word word,word word") == ["word", "word", "word", "word"]
