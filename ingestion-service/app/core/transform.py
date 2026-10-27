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


# A deliberately conservative baseline, not a general language classifier.
_WORDS = {
    "en": set("the and with from this that are was for has have will its company shares market earnings reports profits investors stocks growth revenue".split()),
    "fr": set("le la les des une un et dans pour avec est sont entreprise marché actions bénéfices revenus hausse investisseurs".split()),
    "es": set("el la los las una un y del para con es son empresa mercado acciones beneficios ingresos inversores crecimiento".split()),
}


def detect_language(text: str) -> str:
    words = re.findall(r"[^\W\d_]+", text.casefold())
    if len(words) < 12:
        return "und"
    scores = sorted(((len(set(words) & markers), code) for code, markers in _WORDS.items()), reverse=True)
    best, code = scores[0]
    return code if best >= 5 and best >= scores[1][0] + 3 else "und"
