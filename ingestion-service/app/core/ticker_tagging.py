import re


def tag_tickers(text: str, company_tickers: dict[str, str]) -> list[str]:
    found = {m.upper() for m in re.findall(r"(?<![\w$])\$([A-Za-z][A-Za-z0-9]{0,9}(?:[.-][A-Za-z0-9]{1,3})?)(?![\w.-])", text)}
    for company, ticker in company_tickers.items():
        # Short single-word aliases (IT, ON, A) are too ambiguous without NER.
        if len(company.split()) == 1 and len(company) <= 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(company) + r"(?!\w)", text, re.IGNORECASE):
            found.add(ticker.upper())
    return sorted(found)
