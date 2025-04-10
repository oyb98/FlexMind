import os
import json
import ujson

from functools import partial
from colbert.utils.utils import print_message
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer, tensorize_triples

from colbert.utils.runs import Run


class EagerBatcher():
    def __init__(self, args, rank=0, nranks=1):
        self.rank, self.nranks = rank, nranks
        self.bsize, self.accumsteps = args.bsize, args.accumsteps

        self.query_tokenizer = QueryTokenizer(args.query_maxlen)
        self.doc_tokenizer = DocTokenizer(args.doc_maxlen)
        self.tensorize_triples = partial(tensorize_triples, self.query_tokenizer, self.doc_tokenizer)

        self.triples_path = args.triples
        self.query_maxlen = args.query_maxlen
        self._reset_triples()

    def _reset_triples(self):
        self.reader = open(self.triples_path, mode='r', encoding="utf-8")
        self.position = 0
        self.q2kw = json.load(open("./data/MSMARCO/q2kw_test.json", "r"))

    def __iter__(self):
        return self

    def __next__(self):
        queries, positives, negatives, query_negs, kw_list = [], [], [], [], []

        for line_idx, line in zip(range(self.bsize * self.nranks), self.reader):
            if (self.position + line_idx) % self.nranks != self.rank:
                continue

            query, pos, neg, query_neg = line.strip().split('\t')

            queries.append(query)
            positives.append(pos)
            negatives.append(neg)
            query_negs.append(query_neg)
            if query in self.q2kw.keys():
                kw_list.append(self.q2kw[query])
            else:
                kw_list.append([1]*self.query_maxlen)
            
        self.position += line_idx + 1

        if len(queries) < self.bsize:
            raise StopIteration

        return self.collate(queries, positives, negatives, query_negs, kw_list)

    def collate(self, queries, positives, negatives, query_negs, kw_list):
        assert len(queries) == len(positives) == len(negatives) == len(query_negs) == len(kw_list) == self.bsize

        return self.tensorize_triples(queries, positives, negatives, query_negs, kw_list, self.bsize // self.accumsteps)

    def skip_to_batch(self, batch_idx, intended_batch_size):
        self._reset_triples()

        Run.warn(f'Skipping to batch #{batch_idx} (with intended_batch_size = {intended_batch_size}) for training.')

        _ = [self.reader.readline() for _ in range(batch_idx * intended_batch_size)]

        return None
