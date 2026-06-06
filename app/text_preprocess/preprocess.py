from app.text_preprocess.preprocess_funcs import (lemmatize_text, lower_text,
                                                  normalize_unicode,
                                                  rem_extra_spaces,
                                                  rem_special_chars,
                                                  rem_stop_words)


class Preprocess:
    def __init__(self):
        self.steps = []

    def add_step(self, func):
        self.steps.append(func)
        return self

    def process(self, text: str):
        for step in self.steps:
            text = step(text)
        return text


def vectorizer_preprocess(text: str):
    preprocess = Preprocess()

    preprocess = (
        preprocess.add_step(lower_text)
        .add_step(rem_special_chars)
        .add_step(normalize_unicode)
        .add_step(rem_extra_spaces)
        .add_step(lemmatize_text)
    )
    text = preprocess.process(text)
    return text


def embed_preprocess(text: str):
    preprocess = Preprocess()

    preprocess = (
        preprocess.add_step(lower_text)
        .add_step(rem_special_chars)
        .add_step(normalize_unicode)
        .add_step(rem_extra_spaces)
        .add_step(lemmatize_text)
        .add_step(rem_stop_words)
    )
    text = preprocess.process(text)
    return text
