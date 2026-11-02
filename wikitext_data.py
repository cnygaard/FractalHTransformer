#!/usr/bin/env python3
"""
WikiText helpers shared by training and tokenizer training
"""

import re

WIKITEXT_REPO = 'Salesforce/wikitext'
WIKITEXT_TOKENIZER = 'fractal_gpt_tokenizer_wikitext_8k.json'

# WikiText is pre-tokenized: "1 @,@ 000", "word , word", "game 's", "( text )"
_WIKI_AT_MARKER = re.compile(r' @([-,.])@ ')
_WIKI_SPACE_BEFORE = re.compile(r" (?=[,;:!?)\]]|\.(?!\d)|'s\b|n't\b|'(?=\s))")
_WIKI_SPACE_AFTER = re.compile(r'(?<=[(\[]) ')


def detokenize_wikitext(text):
    """Undo WikiText's punctuation spacing so it matches natural text"""
    text = _WIKI_AT_MARKER.sub(r'\1', text)
    text = _WIKI_SPACE_BEFORE.sub('', text)
    return _WIKI_SPACE_AFTER.sub('', text)


def wikitext_paragraphs(rows):
    """Yield detokenized non-blank WikiText rows (paragraphs and headings), newline-terminated"""
    for example in rows:
        text = detokenize_wikitext(example.get('text', '')).strip()
        if text:
            yield text + '\n'


def wikitext_windows(tokenizer, rows, length):
    """Concatenate WikiText rows into one token stream and cut it into length-token windows"""
    buffer = []
    for text in wikitext_paragraphs(rows):
        buffer.extend(tokenizer.encode(text).ids)
        while len(buffer) >= length:
            yield buffer[:length]
            del buffer[:length]
