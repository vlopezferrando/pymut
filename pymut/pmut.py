# -*- coding: utf-8 -*-

import logging

from .data import get_protein_sequences
from .utils import variant_str


def get_sequences(variants):
    protein_ids = set(variants.protein_id)
    sequences = get_protein_sequences(protein_ids)
    variants['sequence'] = [
        sequences[row.protein_id] if row.protein_id in sequences else None
        for i, row in variants.iterrows()
    ]

    mask = []
    for i, variant in variants.iterrows():
        if variant.position > len(variant.sequence):
            logging.warning(
                'Variant %s out of sequence. Sequence length: %d',
                variant_str(variant), variant.position)
            mask.append(False)
        elif variant.sequence[variant.position - 1] != variant.wt:
            logging.warning(
                'Variant %s does not match sequence. WT should be %s',
                variant_str(variant), variant.sequence[variant.position - 1])
            mask.append(False)
        else:
            mask.append(True)

    return variants[mask]
