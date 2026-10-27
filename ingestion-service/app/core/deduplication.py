import hashlib
import unicodedata


def content_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFC", text).split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class Deduplicator:
    """Request-local exact matches, first accepted document wins."""
    def __init__(self):
        self.urls = set()
        self.hashes = set()

    def duplicate(self, url: str | None, text: str) -> bool:
        digest = content_hash(text)
        if (url and url in self.urls) or digest in self.hashes:
            return True
        if url:
            self.urls.add(url)
        self.hashes.add(digest)
        return False
