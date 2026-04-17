from collections import defaultdict
import unicodedata

import numpy as np
import torch


class TrieTokenizer:
    def __init__(self, vocab, separator=""):
        self.vocab = vocab
        self.separator = separator
        self.trie = self._make_trie()

    def _make_trie(self):
        trie = {}
        for token in self.vocab:
            self._add_token(trie, token)
        return trie

    def _add_token(self, trie, token):
        node = trie
        for char in token:
            if char not in node:
                node[char] = {}
            node = node[char]
        node[""] = token

    def tokenize(self, text):
        tokens = []
        i = 0
        while i < len(text):
            token, length = self._get_next_token(text, i)
            if token:
                tokens.append(token)
                i += length
            else:
                i += 1
        return tokens

    def _get_next_token(self, text, start):
        node = self.trie
        last_token = None
        last_length = 0

        for i, char in enumerate(text[start:]):
            if char not in node:
                break
            node = node[char]
            if "" in node:
                last_token = node[""]
                last_length = i + 1

        return last_token, last_length


class GraphemeTokenizer:
    def __init__(
        self,
        tokenizer_class,
        max_len=64,
        separator="",
        blank_token="_",
        oov_token="▁",
        normalize_unicode=False,
        normalization_mode="NFKC",
        normalizer="unicode",
        printer=print,
        bos_token="_",
        eos_token="_",
        add_bos_token=True,
        add_eos_token=True,
    ):
        self.vocab = list(dict.fromkeys([oov_token, blank_token, bos_token, eos_token]))
        self.max_len = max_len
        self.oov_token = oov_token
        self.blank_token = blank_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.add_bos_token = add_bos_token
        self.add_eos_token = add_eos_token
        self.tokenizer_class = tokenizer_class
        self.separator = separator
        self.tokenizer = self.tokenizer_class([(idx) for idx in self.vocab], separator=self.separator)
        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        self.normalize_unicode = normalize_unicode
        self.normalization_mode = normalization_mode
        self.print = printer
        self.out_of_vocabulary_info = defaultdict(set)
        self.frequency_counter = defaultdict(int)
        self._set_normalizer(normalizer)
        self.pad_token_id = self.word2index[self.blank_token]
        self.bos_token_id = self.word2index[self.bos_token]
        self.eos_token_id = self.word2index[self.eos_token]

    def tokenize(self, text, padding=False, normalize_unicode=None, normalization_mode=None):
        if isinstance(text, list):
            return [self.tokenize(_text, padding, normalize_unicode, normalization_mode) for _text in text]

        normalize_unicode = self.normalize_unicode if normalize_unicode is None else normalize_unicode
        if normalize_unicode:
            text = self._unicode_normalizer(text, normalization_mode)

        tokens = self.tokenizer.tokenize(text)

        if self.add_bos_token and self.add_eos_token:
            tokens = [self.bos_token] + tokens + [self.eos_token]
        elif self.add_bos_token:
            tokens = [self.bos_token] + tokens
        elif self.add_eos_token:
            tokens = tokens + [self.eos_token]

        tokens = tokens[: self.max_len]
        n_tokens = len(tokens)

        if padding:
            tokens = tokens + [self.blank_token] * (self.max_len - n_tokens)

        tokens_id = [self.word2index.get(token, self.word2index[self.oov_token]) for token in tokens]
        attention_mask = [1] * n_tokens + [0] * (len(tokens) - n_tokens)

        return {
            "tokens": tokens,
            "input_ids": tokens_id,
            "token_len": n_tokens,
            "attention_mask": attention_mask,
        }

    def add_tokens(self, vocab, normalize_unicode=None, reset_oov=False):
        normalize_unicode = self.normalize_unicode if normalize_unicode is None else normalize_unicode
        vocab = self._validate_tokens(vocab, normalize_unicode)
        self.vocab = self.vocab + vocab
        self.tokenizer = self.tokenizer_class([(v) for v in self.vocab], separator=self.separator)
        self.word2index = {token: idx for idx, token in enumerate(self.vocab)}
        self.bos_token_id = self.word2index[self.bos_token]
        self.eos_token_id = self.word2index[self.eos_token]
        if reset_oov:
            self.reset_out_of_vocabulary_info(keys=vocab)

    def _validate_tokens(self, vocab, normalize_unicode=False):
        if normalize_unicode:
            vocab = list(map(self._unicode_normalizer, vocab))
        vocab = sorted(list(set(vocab)))
        vocab = [v for v in vocab if v not in self.vocab]
        return vocab

    def _set_normalizer(self, type="unicode"):
        if type == "unicode":
            self.normalizer = lambda text, mode: unicodedata.normalize(mode, text)
        else:
            self.normalizer = lambda text, mode: text

    def _unicode_normalizer(self, text, mode=None):
        mode = self.normalization_mode if mode is None else mode
        text = self.normalizer(text, mode)
        text = text.replace("\u200c", "").replace("\u200d", "")
        return text

    def ids_to_token(self, ids):
        if not ids:
            raise ValueError("ids must be non-empty")
        if not isinstance(ids[0], list):
            token_list = [
                self.vocab[idx]
                for idx in ids
                if self.vocab[idx] not in [self.blank_token, self.bos_token, self.eos_token]
            ]
            return token_list
        if isinstance(ids[0], list):
            return list(map(self.ids_to_token, ids))

    def ids_to_text(self, ids):
        if not ids:
            raise ValueError("ids must be non-empty")
        tokens = self.ids_to_token(ids)
        if not isinstance(tokens[0], list):
            return "".join(tokens)
        if isinstance(tokens[0], list):
            return list(map("".join, tokens))

    def reset_out_of_vocabulary_info(self, keys=None):
        if isinstance(keys, list):
            for key in keys:
                self.out_of_vocabulary_info.pop(key, None)
            return
        if isinstance(keys, str):
            if keys.lower() == "all":
                self.out_of_vocabulary_info = defaultdict(set)
            return


class BnGraphemizerProcessor:
    def __init__(
        self,
        grapheme_file,
        model_max_length=128,
        normalize_unicode=True,
        normalization_mode="NFKC",
        normalizer="unicode",
        blank_token="_",
        bos_token="<s>",
        eos_token="</s>",
        add_bos_token=True,
        add_eos_token=True,
    ):
        self.grapheme_file = grapheme_file
        self.model_max_length = model_max_length
        self.blank_token = blank_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.list_of_graphemes = self._load_graphemes()
        self.bn_graphmemizer = self._initialize_graphemizer()
        self.pad_token_id = self.bn_graphmemizer.pad_token_id
        self.bos_token_id = self.bn_graphmemizer.bos_token_id
        self.eos_token_id = self.bn_graphmemizer.eos_token_id
        self.vocab = self.bn_graphmemizer.vocab

    def _load_graphemes(self):
        with open(self.grapheme_file, "r", encoding="utf-8") as file:
            graphemes = sorted(list(set([line.rstrip("\n\r") for line in file.readlines() if line.strip()])))
        return graphemes

    def _initialize_graphemizer(self):
        graphemizer = GraphemeTokenizer(
            tokenizer_class=TrieTokenizer,
            max_len=self.model_max_length,
            blank_token=self.blank_token,
            bos_token=self.bos_token,
            eos_token=self.eos_token,
            add_bos_token=True,
            add_eos_token=True,
        )
        graphemizer.add_tokens(self.list_of_graphemes, reset_oov=True)
        return graphemizer

    def __call__(self, texts, padding=False):
        bng_text_inputs = self.bn_graphmemizer.tokenize(texts, padding=padding)
        bng_inputs = self._get_tokenized_inputs(bng_text_inputs)
        bng_input_ids = torch.Tensor(bng_inputs["input_ids"]).long()
        bng_attention_mask = torch.Tensor(bng_inputs["attention_mask"]).long()
        if bng_input_ids.ndim == 1:
            bng_input_ids = bng_input_ids.unsqueeze(0)
        if bng_attention_mask.ndim == 1:
            bng_attention_mask = bng_attention_mask.unsqueeze(0)
        return {"input_ids": bng_input_ids, "attention_mask": bng_attention_mask}

    def _get_tokenized_inputs(self, inputs):
        if not isinstance(inputs, list):
            return {"input_ids": inputs["input_ids"], "attention_mask": inputs["attention_mask"]}
        input_ids, attention_mask = [], []
        for item in inputs:
            if isinstance(item, list):
                item = self._get_tokenized_inputs(item)
            input_ids.append(item["input_ids"])
            attention_mask.append(item["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, input_ids):
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy() if input_ids.is_cuda else input_ids.numpy()
        if isinstance(input_ids, np.ndarray):
            input_ids = input_ids.tolist()
        if isinstance(input_ids, list):
            if len(input_ids) == 0:
                return ""
            if isinstance(input_ids[0], list):
                return [self.decode(ids) for ids in input_ids]
            token_list = self.bn_graphmemizer.ids_to_token(input_ids)
            return "".join(token_list)

        raise ValueError("Unsupported input_ids type")
