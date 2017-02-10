import re

import numpy as np
import pandas as pd

from .amino_acids import AMINO_ACIDS, AA_DISTANCE, aa1


def variants():
    return pd.DataFrame(columns=[
        'protein_id', 'sequence', 'position', 'wt', 'mt', 'disease'])


def variant_str(variant):
    return '%s %d.%s>%s' % (
        variant.protein_id, variant.position, variant.wt, variant.mt)


def load_fasta_str(f):
    return {
        descr: b''.join(sequence).decode()
        for descr, sequence in load_fasta_iter(f)
    }


def load_fasta_iter(f):
    descr = ''
    sequence = b''

    for line in f:
        line = line.strip()
        if line == b'':
            continue
        if line[0] == ord(b'>'):
            if descr:
                yield (descr, np.array([chr(c) for c in sequence],
                       dtype=np.character))
                descr = ''
                sequence = ''
            descr = line[1:].decode()
            sequence = b''
        else:
            sequence += line

    if descr:
        yield (descr, np.array([chr(c) for c in sequence], dtype=np.character))


def all_variants(protein_id, sequence, position=None):
    """Position is 0-based"""
    positions = [(position, sequence[position])] \
        if position else enumerate(sequence)

    return pd.DataFrame([
            [protein_id, sequence, i+1, wt, mt]
            for i, wt in positions
            for mt in AMINO_ACIDS
            if mt != wt
        ], columns=['protein_id', 'sequence', 'position', 'wt', 'mt'])


def parse_variant(string):
    p = re.compile(r'p.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2})')
    m = p.match(string)

    assert m and m.group() == string, '%s cannot be parsed as variant' % string

    return {
        'position': int(m.group(2)),
        'wt': aa1(m.group(1)),
        'mt': aa1(m.group(3)),
    }


def position_score(wt, position_scores):
    scores_at_dist_1 = [
        sc
        for mt, sc in position_scores.items()
        if sc and AA_DISTANCE[wt][mt] == 1
    ]
    if not scores_at_dist_1:
        return np.median(list(position_scores.values()))

    return np.median(scores_at_dist_1) if scores_at_dist_1 else None
