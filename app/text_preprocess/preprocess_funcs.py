import re
import unicodedata
from concurrent.futures import ProcessPoolExecutor

from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from app.models.chunk import Chunk

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

"""
Required NLTK resources

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("wordnet")
nltk.download("omw-1.4")
nltk.download("stopwords")
"""


def _get_wordnet_pos(tag: str):
    """
    Convert NLTK POS tags to WordNet POS tags.
    """
    if tag.startswith("J"):
        return wordnet.ADJ

    if tag.startswith("V"):
        return wordnet.VERB

    if tag.startswith("N"):
        return wordnet.NOUN

    if tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


def lemmatize_text(text: str) -> str:
    """
    Lemmatize a text using POS tagging.
    """
    words = word_tokenize(text)
    tagged_words = pos_tag(words)
    lemmas = [
        lemmatizer.lemmatize(word, _get_wordnet_pos(tag)) for word, tag in tagged_words
    ]
    return " ".join(lemmas)


def lower_text(text: str) -> str:
    return text.lower()


def rem_extra_spaces(text: str) -> str:
    return " ".join(text.split())


def rem_special_chars(text: str) -> str:
    return re.sub(r"[^\w\s.+#/-]", " ", text)


def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def rem_stop_words(text: str) -> str:
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOP_WORDS]
    return " ".join(filtered_words)
