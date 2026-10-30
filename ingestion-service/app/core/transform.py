import html
from html.parser import HTMLParser
import re
import unicodedata
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.rsplit(":", 1)[-1]
        if tag in {"script", "style"}:
            self.hidden += 1
        if tag in {"p", "div", "br", "li", "section", "h1", "h2"}:
            self.parts.append(" ")

    def handle_endtag(self, tag):
        tag = tag.rsplit(":", 1)[-1]
        if tag in {"script", "style"}:
            self.hidden = max(0, self.hidden - 1)
        if tag in {"p", "div", "li", "section", "h1", "h2"}:
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def clean_text(value: str | None) -> str:
    parser = TextParser()
    parser.feed(value or "")
    parser.close()
    return " ".join(unicodedata.normalize("NFC", html.unescape("".join(parser.parts))).split())


def canonical_url(value: str | None) -> str | None:
    if not value:
        return None
    try:
        parts = urlsplit(value.strip())
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return None
        host = parts.hostname.lower().encode("idna").decode()
        if ":" in host:
            host = "[" + host + "]"
        port = parts.port
        if port and (parts.scheme.lower(), port) not in {("http", 80), ("https", 443)}:
            host += f":{port}"
        query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                 if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}]
        return urlunsplit((parts.scheme.lower(), host, parts.path or "/", urlencode(query), ""))
    except (ValueError, UnicodeError):
        return None


# Small deterministic vocabularies, not a general language classifier.
_WORDS = {
    "en": set("the and with from this that are was for has have will its company shares market earnings reports profits investors stocks growth revenue".split()),
    "fr": set("le la les des une un et dans pour avec est sont entreprise marché actions bénéfices revenus hausse investisseurs".split()),
    "es": set("el la los las una un y del para con es son empresa mercado acciones beneficios ingresos inversores crecimiento".split()),
}

_HEADLINE_WORDS = {
    "en": set("quarterly annual results rises rise falls fall stocks stock earnings profits revenue shares investors company growth reports reported forecast outlook cuts beats after amid raises increased declines publishes announces surged surged yields bond bonds rates inflation bank banks".split()),
    "fr": set("à aux du au ce cette ses sur après malgré trimestre trimestriels résultats résultat bourse boursière boursier bénéfice bénéfices chiffre affaires hausse baisse augmente augmente annonce publie prévisions taux banque banques économie croissance investisseurs marché revenus".split()),
    "es": set("al sus por tras pese trimestre trimestrales resultados bolsa bursátil beneficio beneficios ganancias aumenta sube caen caída bajan anuncia publica previsiones tipos bancos economía crecimiento inversores mercado ingresos".split()),
}
_ALL_MARKERS = {code: markers | _HEADLINE_WORDS[code] for code, markers in _WORDS.items()}
# Shared words such as "la", "un", "con" in other contexts, or "trimestre"
# alone cannot establish a language. Only language-specific evidence votes.
_UNIQUE_MARKERS = {
    code: markers - set().union(*(other for language, other in _ALL_MARKERS.items() if language != code))
    for code, markers in _ALL_MARKERS.items()
}
_KNOWN_LANGUAGE_WORDS = set().union(*_ALL_MARKERS.values())
_LANGUAGE_TOKENS = re.compile(r"[^\W\d_]+")
_NON_LANGUAGE = re.compile(r"https?://\S+|\$[A-Za-z][A-Za-z0-9.\-]*", re.IGNORECASE)


def detect_language(text: str) -> str:
    """Identify en/fr/es from distinct lexical evidence, otherwise return und.

    Require three tokens and two distinct exclusive markers with a margin of
    two. Comparable evidence for another language means mixed/uncertain text.
    No translation, probabilistic confidence, model loading or network access.
    """
    normalized = unicodedata.normalize("NFC", text).casefold()
    words = _LANGUAGE_TOKENS.findall(_NON_LANGUAGE.sub(" ", normalized))
    if len(words) < 3:
        return "und"
    unique = set(words)
    scores = sorted(((len(unique & markers), code) for code, markers in _UNIQUE_MARKERS.items()), reverse=True)
    best, code = scores[0]
    runner_up = scores[1][0]
    if best < 2 or best - runner_up < 2 or (runner_up >= 2 and runner_up >= best * 0.45):
        return "und"
    # Isolated familiar words inside predominantly unknown text are unreliable.
    known = len(unique & _KNOWN_LANGUAGE_WORDS)
    if known / len(unique) < 0.2:
        return "und"
    return code
