import re
from encodings import utf_8
import pandas as pd
import numpy as np

from tqdm import tqdm
from math import ceil
from functools import reduce
from collections import defaultdict
from warnings import warn
from pathlib import Path
from multiprocessing import Pool, cpu_count, Lock
from pudb import set_trace


class TextCoder:
    def __init__(self, corpus, vocabulary_size=2000):
        self.counts = (corpus.str.split(" ", expand=True).stack().value_counts()).head(
            vocabulary_size - 3
        )
        self.vocabulary = sorted(
            list(set(["_START_", "_STOP_", "_OTHER_"]).union(set(self.counts.index)))
        )
        self.word2int = defaultdict(lambda: self.vocabulary.index("_OTHER_"))
        self.int2word = {}
        for i, w in enumerate(self.vocabulary):
            self.word2int[w] = i
            self.int2word[i] = w

    def encode(self, data, one_hot=False):
        result = []
        for entry in data:
            words = entry.split(" ")
            if "_START_" != words[0]:
                words = ["_START_"] + words
            if "_STOP_" != words[-1]:
                words = words + ["_STOP_"]
            vector = np.zeros((len(words), len(self.vocabulary) if one_hot else 1))
            for i, w in enumerate(words):
                if one_hot:
                    vector[i, self.word2int[w]] = 1
                else:
                    vector[i, 0] = self.word2int[w]
            result.append(vector)
        return result

    def decode(self, data, one_hot=False):
        result = []
        for entry in data:
            sentence = []
            for i in range(entry.shape[0]):
                sentence.append(
                    self.int2word[entry[i, :].argmax() if one_hot else entry[i, 0]]
                )
            result.append(" ".join(sentence))
        return result


def preprocess_text(series: pd.Series):
    series = series.apply(clear_parantheses)
    series = series.str.lower()
    series = series.str.replace(r"[^a-z\d\såäö]", "")
    series = series.str.replace(r"[0-9]+\.[0-9]+", "_FLOAT_ ")
    series = series.str.replace(r"[0-9]+", "_INTEGER_ ")
    series = series.str.replace(r"\s+", " ")
    series = series.str.strip()
    return series


def clear_parantheses(line: str):
    allow_unbalanced = False
    result = ""
    level = 0
    for c in line:
        if c == "(":
            level += 1
        elif c == ")":
            level -= 1
        elif level == 0:
            result += c
    if level == 0 or allow_unbalanced:
        return result
    else:
        return None


def load_file(path: Path):
    with open(path) as file:
        lines = file.read().split("\n")
    return lines


def load_data(paths: list, names: list, chunksize=10000, quiet=False):
    data = pd.DataFrame.from_dict(
        {name: load_file(path) for name, path in zip(names, paths)}, orient="columns"
    )

    for name in names:
        data[name] = preprocess_text(data[name])

    failed = (data[names] == "").any(axis=1)
    num_failed = sum(failed)
    if num_failed > 0:
        if not quiet:
            warning_string = f"%d entries dropped" % num_failed
            warn(warning_string)
        data = data[failed == False]
    return data.dropna()
