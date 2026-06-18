from concurrent.futures import ProcessPoolExecutor

from app.models.chunk import Chunk
from app.text_preprocess.preprocess_funcs import (
    lemmatize_text,
    lower_text,
    normalize_unicode,
    rem_extra_spaces,
    rem_special_chars,
    rem_stop_words,
)


def preprocess_pipeline(chunk: Chunk | str):
    text = chunk.text if isinstance(chunk, Chunk) else chunk
    text = lower_text(text)
    text = rem_special_chars(text)
    text = normalize_unicode(text)
    text = rem_extra_spaces(text)
    text = lemmatize_text(text)
    return {"embed": text, "vector": rem_stop_words(text)}


def preprocess_text(chunks: list[Chunk] | list[str]) -> dict[str, list[str]]:
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(preprocess_pipeline, chunks))
    cleaned = {"vectors": [], "embeds": []}
    for res in results:
        cleaned["vectors"].append(res["vector"])
        cleaned["embeds"].append(res["embed"])
    return cleaned
