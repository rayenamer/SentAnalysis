"""Bounded RSS/Atom extraction. Never fetch article links or follow redirects."""
from dataclasses import dataclass, field
import xml.etree.ElementTree as ET

import httpx
from app.helper.provider_http import SourceError, fetch_bytes


class SafeTreeBuilder(ET.TreeBuilder):
    def __init__(self):
        super().__init__()
        self.depth = 0

    def start(self, tag, attrs):
        self.depth += 1
        if self.depth > 128:
            raise SourceError("unsafe_xml")
        return super().start(tag, attrs)

    def end(self, tag):
        result = super().end(tag)
        self.depth -= 1
        return result

    def doctype(self, name, pubid, system):
        raise SourceError("unsafe_xml")


@dataclass
class RawDocument:
    title: str | None
    text: str | None
    url: str | None
    published_at: str | None
    metadata: dict = field(default_factory=dict)


def local_name(tag):
    return tag.rsplit("}", 1)[-1]


def parse_feed(data: bytes, limit: int) -> tuple[list[RawDocument], bool]:
    try:
        root = ET.fromstring(data, parser=ET.XMLParser(target=SafeTreeBuilder()))
    except (ET.ParseError, ValueError, LookupError) as exc:
        raise SourceError("invalid_feed") from exc
    kind = local_name(root.tag)
    if kind not in {"rss", "feed", "RDF"}:
        raise SourceError("invalid_feed")
    records = []
    for entry in root.iter():
        if local_name(entry.tag) not in {"item", "entry"}:
            continue
        if len(records) == limit:
            return records, True
        fields = {}
        link = None
        for child in entry:
            name = local_name(child.tag)
            # Preserve inline/block boundaries in Atom XHTML; cleaning strips tags.
            fields[name] = ((child.text or "") + "".join(
                ET.tostring(nested, encoding="unicode") for nested in child)).strip()
            if name == "link":
                if kind == "feed":
                    if child.get("rel", "alternate") == "alternate" and child.get("href"):
                        link = child.get("href")
                elif fields[name]:
                    link = fields[name]
        records.append(RawDocument(fields.get("title"), fields.get("encoded") or fields.get("content") or fields.get("description") or fields.get("summary"), link,
                                   fields.get("pubDate") or fields.get("published") or fields.get("date") or fields.get("updated")))
    return records, False


class RSSExtractor:
    def __init__(self, client: httpx.AsyncClient, settings):
        self.client = client
        self.settings = settings

    async def fetch(self, source):
        data = await fetch_bytes(self.client, "GET", source.url, timeout=self.settings.timeout_seconds,
                                 max_bytes=self.settings.max_response_bytes)
        return parse_feed(data, self.settings.max_entries_per_source)
