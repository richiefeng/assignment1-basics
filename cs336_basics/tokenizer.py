"""
Simple BPE Tokenizer implementation for the training script.
This is a basic implementation that can be extended as needed.
"""

import json
import regex as re
from typing import List, Dict, Tuple


class BPETokenizer:
    """Basic BPE Tokenizer implementation."""

    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Create reverse mapping from bytes to token ID
        self.token_to_id = {
            token_bytes: token_id for token_id, token_bytes in vocab.items()
        }

        # Add special tokens to vocab if they don't exist
        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in self.token_to_id:
                # Find the next available ID
                next_id = max(vocab.keys()) + 1 if vocab else 0
                self.vocab[next_id] = special_bytes
                self.token_to_id[special_bytes] = next_id

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if not text:
            return []

        # First, split text by special tokens to preserve them
        segments = self._split_by_special_tokens(text)

        token_ids = []
        for segment in segments:
            if segment in self.special_tokens:
                # This is a special token
                special_bytes = segment.encode("utf-8")
                if special_bytes in self.token_to_id:
                    token_ids.append(self.token_to_id[special_bytes])
            else:
                # This is regular text, apply GPT-2 regex pattern
                PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
                pattern = re.compile(PAT)

                for match in pattern.finditer(segment):
                    pretoken = match.group()
                    if not pretoken:
                        continue

                    # Encode the pretoken using BPE
                    pretoken_ids = self._encode_pretoken(pretoken)
                    token_ids.extend(pretoken_ids)

        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text."""
        if not token_ids:
            return ""

        # Convert token IDs to bytes
        token_bytes_list = []
        for token_id in token_ids:
            if token_id in self.vocab:
                token_bytes_list.append(self.vocab[token_id])
            else:
                # Handle unknown token IDs
                token_bytes_list.append(b"")

        # Concatenate all bytes
        all_bytes = b"".join(token_bytes_list)

        # Decode to string, handling invalid UTF-8
        try:
            return all_bytes.decode("utf-8")
        except UnicodeDecodeError:
            return all_bytes.decode("utf-8", errors="replace")

    def _split_by_special_tokens(self, text: str) -> List[str]:
        """Split text by special tokens while preserving the special tokens."""
        if not self.special_tokens:
            return [text]

        # Sort special tokens by length (longest first) to handle overlapping tokens
        sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)

        # Find all occurrences of special tokens in the text
        matches = []
        for special_token in sorted_special_tokens:
            start = 0
            while True:
                pos = text.find(special_token, start)
                if pos == -1:
                    break
                matches.append((pos, pos + len(special_token), special_token))
                start = pos + 1

        # Sort matches by position
        matches.sort(key=lambda x: x[0])

        # Remove overlapping matches (keep the longest one when there's overlap)
        filtered_matches = []
        for match in matches:
            start, end, token = match
            # Check if this match overlaps with any existing match
            overlaps = False
            for existing_start, existing_end, _ in filtered_matches:
                if not (end <= existing_start or start >= existing_end):
                    overlaps = True
                    break
            if not overlaps:
                filtered_matches.append(match)

        # Sort matches by position again
        filtered_matches.sort(key=lambda x: x[0])

        # Split text based on matches
        segments = []
        last_end = 0

        for start, end, token in filtered_matches:
            # Add text before the special token
            if start > last_end:
                segments.append(text[last_end:start])
            # Add the special token
            segments.append(token)
            last_end = end

        # Add remaining text after the last special token
        if last_end < len(text):
            segments.append(text[last_end:])

        return segments

    def _encode_pretoken(self, pretoken: str) -> List[int]:
        """Encode a single pretoken using BPE."""
        # Convert pretoken to bytes
        pretoken_bytes = pretoken.encode("utf-8")

        # Start with individual bytes
        current_tokens = [bytes([b]) for b in pretoken_bytes]

        # Apply BPE merges in order
        for token1, token2 in self.merges:
            merged = token1 + token2

            # Keep applying this merge until no more can be applied
            while True:
                new_tokens = []
                i = 0
                merged_this_round = False

                while i < len(current_tokens):
                    if (
                        i < len(current_tokens) - 1
                        and current_tokens[i] == token1
                        and current_tokens[i + 1] == token2
                    ):
                        new_tokens.append(merged)
                        i += 2
                        merged_this_round = True
                    else:
                        new_tokens.append(current_tokens[i])
                        i += 1

                current_tokens = new_tokens

                # If no merges were applied this round, move to next merge
                if not merged_this_round:
                    break

        # Convert final tokens to IDs
        token_ids = []
        for token_bytes in current_tokens:
            if token_bytes in self.token_to_id:
                token_ids.append(self.token_to_id[token_bytes])
            else:
                # Handle unknown tokens by encoding as individual bytes
                for b in token_bytes:
                    byte_token = bytes([b])
                    if byte_token in self.token_to_id:
                        token_ids.append(self.token_to_id[byte_token])

        return token_ids


def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: List[str] = None,
) -> BPETokenizer:
    """Factory function to create a BPE tokenizer."""
    return BPETokenizer(vocab, merges, special_tokens)

