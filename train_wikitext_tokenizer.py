#!/usr/bin/env python3
"""
Train a byte-level BPE tokenizer on WikiText-103 for --wikitext training

Byte-level BPE keeps spaces and newlines and never produces [UNK], unlike the
TinyStories tokenizer, which drops whitespace and has no byte fallback.
"""

import argparse

from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from wikitext_data import WIKITEXT_REPO, WIKITEXT_TOKENIZER, wikitext_paragraphs

# Same special tokens (and ids) as the TinyStories tokenizer
SPECIAL_TOKENS = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', '[EOS]']


def main():
    parser = argparse.ArgumentParser(description='Train the WikiText-103 tokenizer')
    parser.add_argument('--vocab-size', type=int, default=8000, help='Vocabulary size (must match the model vocab_size)')
    parser.add_argument('--output', type=str, default=WIKITEXT_TOKENIZER, help='Output tokenizer file')
    args = parser.parse_args()

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    rows = load_dataset(WIKITEXT_REPO, 'wikitext-103-raw-v1', split='train', streaming=True)
    tokenizer.train_from_iterator(wikitext_paragraphs(rows), trainer=trainer)
    tokenizer.save(args.output)
    print(f"Saved {tokenizer.get_vocab_size()}-token tokenizer to {args.output}")


if __name__ == '__main__':
    main()
