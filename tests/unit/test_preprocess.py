from app.preprocess import (
    rem_extra_spaces,
    rem_special_chars,
    rem_stop_words,
    lower_text,
    normalize_unicode,
    lemmatize_text,
)


def test_rem_special_chars():
    text = "c++ node.js, c#, lua, ##, ,, $$1232##"
    text = rem_special_chars(text)
    assert "," not in text
    assert "$" not in text


def test_rem_stop_words():
    text = "anand is and the anand stop word the"
    text = rem_stop_words(text)
    assert text == "anand anand stop word"


def test_lemmatize_txt():
    text = "anand running, eating, walking"
    text = lemmatize_text(text)
    assert "running" not in text
    assert "eating" not in text
    assert "walking" not in text
