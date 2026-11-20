"""Text-parsing tests for fix B3: inline <think> stripping, non-greedy JSON."""
from bench.scoring import extract_json
from bench.textutil import strip_think


def test_strip_think_then_extract_answer():
    text = '<think>{"answer": 1}\n</think>\n{"answer": 396}'
    assert extract_json(strip_think(text)) == {"answer": 396}


def test_prose_wrapped_json_takes_the_first_valid_object():
    assert extract_json('Sure: {"answer": 5} (see {x})') == {"answer": 5}


def test_non_dict_json_is_none():
    assert extract_json('[1,2]') is None
    assert extract_json('42') is None


def test_unclosed_think_means_no_answer():
    assert strip_think("<think>abc") == ""


def test_plain_text_untouched():
    assert strip_think("hello") == "hello"
