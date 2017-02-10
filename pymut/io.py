
import logging

import pandas as pd

from .utils import parse_variant


def read_humsavar_txt(f):
    """Read a humsavar.txt file and return a variants DataFrame"""
    SWISSPROT_POS, AA_CHANGE_POS, TYPE_POS = (1, 3, 4)

    variants = set()
    header = True

    for line in f:
        line = line.strip()

        # Ignore empty lines
        if line == '':
            continue

        # Log header info
        if header:
            if '________' in line:
                header = False
            if ('Release:' in line or 'Disease variants:' in line or
                    'Polymorphisms:' in line or 'Unclassified variants:' in
                    line or 'Total:' in line):
                logging.info(line)
        else:
            # Break when the end comment is reached
            if '--------------' in line:
                break

            row = line.split()

            try:
                uniprot = row[SWISSPROT_POS]
                variant = parse_variant(row[AA_CHANGE_POS])
                typ = row[TYPE_POS]
                assert typ in ['Disease', 'Polymorphism', 'Unclassified']

                if typ == 'Disease' or typ == 'Polymorphism':
                    variants.add((
                        uniprot, variant['position'],
                        variant['wt'], variant['mt'], typ == 'Disease'))
            except:
                # Print line if error when parsing
                logging.warning('Can not parse line: %s', line)

    return pd.DataFrame(
        list(variants),
        columns=('protein_id', 'position', 'wt', 'mt', 'disease'))
