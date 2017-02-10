# -*- coding: utf-8 -*-

import collections
import gzip
import itertools
import io
import logging
import math
import re
import sys

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import numpy as np
import seaborn as sns

from .amino_acids import _SUBSTITUTION_MATRICES
from .data import path_blast, path_kalign, uniref100, get_network_parameters
from .utils import load_fasta_iter

############
# Features #
############

_DBS = ['100', '90']

_FEATURES = {
    'basic': ['hkd', 'pos'] +
             ['blosum50', 'blosum62', 'blosum80', 'miyata', 'pam60'] +
             [''.join(l) for l in itertools.product(
                 ['hwo', 'hww', 'vol'], ['_d', '_r'])],
    'topology': ['top_' + s for s in [
                 'betweenness', 'cc', 'closeness', 'degree',
                 'degree_centrality', 'eigenvector']],
    'blast': [''.join(l) for l in itertools.product(
              ['b'],
              [d[0] for d in _DBS],
              ['_all', '_eva', '_hum', '_nhu'],
              ['_nwt', '_nmt', '_naa', '_nal', '_rwt', '_rmt', '_pwm'],
              ['', '_w'])],
    'kalign': [''.join(l) for l in itertools.product(
               ['k'],
               [d[0] for d in _DBS],
               ['_all', '_hum', '_nhu'],
               ['_nwt', '_nmt', '_naa', '_nal', '_rwt', '_rmt', '_pwm',
                '_dh', '_dp', '_de', '_dv', '_da', '_db'],
               ['', '_w'])]
}

# Prepend 'feature_'
_FEATURES = {k: ['feature_' + x for x in v] for k, v in _FEATURES.items()}

FEATURES = tuple(itertools.chain(*_FEATURES.values()))

_BLAST_FEATURES = {
    d: set([f for f in _FEATURES['blast'] if f.startswith('feature_b' + d[0])])
    for d in _DBS
}

_KALIGN_FEATURES = {
    d: set([f for f in _FEATURES['kalign']
            if f.startswith('feature_k' + d[0])])
    for d in _DBS
}

PMUT_FEATURES = sorted([
    'feature_k1_all_nwt_w',
    'feature_b1_eva_rwt',
    'feature_k1_all_nal',
    'feature_k9_all_nwt_w',
    'feature_b1_eva_naa',
    'feature_k9_hum_naa_w',
    'feature_b9_eva_naa',
    'feature_k9_all_naa_w',
    'feature_k1_all_pwm_w',
    'feature_b1_eva_pwm_w',
    'feature_k1_hum_naa',
    'feature_miyata',
])

#################
# Feature names #
#################

# TODO: update feature names

_FEATURE_NAMES = {
    'feature_blosum62': 'BLOSUM 62',
    'feature_hwo_d': 'Hidrophobicity difference',
    'feature_top_degree': 'Topology Degree',
    'feature_b1_eva_nwt': 'BLAST UniRef100 # wt',
    'feature_b1_eva_nmt_w': 'BLAST UniRef100 % mt',
    'feature_b1_eva_naa': 'BLAST UniRef100 # aas',
    'feature_b1_eva_pwm': 'BLAST UniRef100 PSSM',
    'feature_b1_eva_pwm_w': 'BLAST UniRef100 weighted PSSM',
    'feature_k1_all_nwt': 'Kalign UniRef100 # wt',
    'feature_k1_all_nal': 'Kalign UniRef100 # align',
    'feature_k1_all_pwm_w': 'Kalign UniRef100 weighted PSSM',
    'feature_k1_hum_nmt_w': 'Kalign Human UniRef100 weighted # mt',
    'feature_k1_hum_nal': 'Kalign Human UniRef100 # align',
    'feature_k9_all_nwt_w': 'Kalign UniRef90 weighted # wt',
    'feature_k9_hum_naa': 'Kalign Human UniRef90',
}


def feature_name(feature):
    """Return a human readable name of the feature"""
    if feature in _FEATURE_NAMES:
        return _FEATURE_NAMES[feature]
    return feature


########################
# Features computation #
########################

def compute_features(variants, features=None):
    """Compute features of variants"""

    if features is None:
        features = FEATURES

    _basic_features(variants, features)
    _topology_features(variants, features)
    _blast_features(variants, features)
    _kalign_features(variants, features)


##################
# Basic features #
##################

# Average volume of buried residues
# Source: Amino Acid Volumes. -C. Chothia, Nature 254:304-308(1975)
_AA_Vol = {
    'A':  92., 'C': 106., 'D': 125., 'E': 155., 'F': 203., 'G':  66.,
    'H': 167., 'I': 169., 'K': 171., 'L': 168., 'M': 171., 'N': 125.,
    'P': 129., 'Q': 161., 'R': 202., 'S':  99., 'T': 122., 'V': 142.,
    'W': 238., 'Y': 204.,
    'U': 0.0, 'X': 0.0  # Invented
}

# Hydrophobicity
# Source: Wimley and White, Nat Struct Biol 3:842 (1996),
#         http://www.cgl.ucsf.edu/chimera/docs/UsersGuide/midas/hydrophob.html
_AA_Hww = {
    "D": -1.23, "E": -2.02, "N": -0.42, "Q": -0.58, "K": -0.99, "R": -0.81,
    "H": -0.96, "G": -0.01, "P": -0.45, "S": -0.13, "T": -0.14, "C":  0.24,
    "M":  0.23, "A": -0.17, "V": -0.07, "I":  0.31, "L":  0.56, "F":  1.13,
    "W":  1.85, "Y":  0.94,
    'U': 0.0, 'X': 0.0  # Invented
}

# Free energies of transfer between Octanol and water
# Source: Octanol and water. By Fauchere & Pliska.(1989)
_AA_Hwo = {
    "A": 0.42, "C":  1.34, "D": -1.05, "E": -0.87, "F": 2.44, "G":  0.00,
    "H": 0.18, "I":  2.46, "K": -1.35, "L":  2.32, "M": 1.68, "N": -0.82,
    "P": 0.98, "Q": -0.30, "R": -1.37, "S": -0.05, "T": 0.35, "V":  1.66,
    "W": 3.07, "Y":  1.31,
    'U': 0.0, 'X': 0.0  # Invented
}


# Hydrophaty Index
# Source: A simple method for displaying the hydropathic character of a
#         protein. Kyte and Doolittle
_AA_Hkd = {
    "F":  2.8, "I":  4.5, "W": -0.9, "L":  3.8, "V":  4.2, "M":  1.9,
    "Y":  6.3, "C":  2.5, "A":  1.8, "T": -1.3, "H": -3.2, "G": -0.4,
    "S": -0.8, "Q": -3.5, "R": -4.5, "K": -3.9, "N": -3.5, "E": -3.5,
    "P": -1.6, "D": -3.5, "X": -0.14,
    'U': 0.0, 'X': 0.0  # Invented
}


def _diff(variant, values_dict):
    return values_dict[variant.wt] - values_dict[variant.mt]


def _rdiff(variant, values_dict):
    diff = _diff(variant, values_dict)
    if values_dict[variant.wt] == 0:
        return diff / sys.float_info.epsilon
    return diff / values_dict[variant.wt]


def _kyte_doolittle(variant, window_size=9):
    start = (variant.position-1) - (window_size - 1)//2
    if start < 0:
        start = 0

    end = (variant.position-1) + (window_size + 1)//2
    if end > len(variant.sequence):
        end = len(variant.sequence)

    sum_wt = sum([_AA_Hkd[aa] for aa in variant.sequence[start:end]])
    sum_mt = sum_wt - _AA_Hkd[variant.wt] + _AA_Hkd[variant.mt]

    return float(sum_wt - sum_mt) / (end - start)


def _matrix(variant, matrix):
    if (variant.wt, variant.mt) in matrix:
        return matrix[(variant.wt, variant.mt)]
    elif (variant.mt, variant.wt) in matrix:
        return matrix[(variant.mt, variant.wt)]
    else:
        return np.nan


def _basic_feature(variants, feature):
    if feature == 'feature_pos':
        return variants['position']
    elif feature == 'feature_hkd':
        return variants.apply(_kyte_doolittle, axis=1)
    elif feature == 'feature_vol_d':
        return variants.apply(_diff, args=(_AA_Vol,), axis=1)
    elif feature == 'feature_vol_r':
        return variants.apply(_rdiff, args=(_AA_Vol,), axis=1)
    elif feature == 'feature_hww_d':
        return variants.apply(_diff, args=(_AA_Hww,), axis=1)
    elif feature == 'feature_hww_r':
        return variants.apply(_rdiff, args=(_AA_Hww,), axis=1)
    elif feature == 'feature_hwo_d':
        return variants.apply(_diff, args=(_AA_Hwo,), axis=1)
    elif feature == 'feature_hwo_r':
        return variants.apply(_rdiff, args=(_AA_Hwo,), axis=1)

    m = re.match('feature_(blosum80|blosum62|blosum50|pam60|miyata)', feature)
    if m:
        matrix = _SUBSTITUTION_MATRICES[m.group(1)]
        return variants.apply(_matrix, args=(matrix,), axis=1)


def _basic_features(variants, features):
    logging.info('Computing basic features...')
    for feature in set(features).intersection(_FEATURES['basic']):
        variants[feature] = _basic_feature(variants, feature)


#####################
# Topology features #
#####################

_TOPOLOGY_FEATURES = {}
_TOPOLOGY_FEATURES_INDEX = {}


def _load_topology_features():
    global _TOPOLOGY_FEATURES
    global _TOPOLOGY_FEATURES_INDEX
    _TOPOLOGY_FEATURES, _TOPOLOGY_FEATURES_INDEX = get_network_parameters()


def _topology_features(variants, features):
    logging.info('Computing topology features...')
    _load_topology_features()
    for feature in set(features).intersection(_FEATURES['topology']):
        variants[feature] = _topology_feature(variants, feature)


def _topology_feature(variants, feature):
    index = _TOPOLOGY_FEATURES_INDEX[feature]
    return variants.apply(_topology_variant_feature, args=(index,), axis=1)


def _topology_variant_feature(variant, index):
    if variant.protein_id in _TOPOLOGY_FEATURES:
        return _TOPOLOGY_FEATURES[variant.protein_id][index]
    return np.nan


##################
# Blast features #
##################

_BLAST_EVALUE = {
    '90': 1e-45,
    '100': 1e-75,
    'test': 1e-75,
}


def _blast_features(variants, features):
    logging.info('Computing blast features for %d variants...' % len(variants))
    for db, blast_features in _BLAST_FEATURES.items():
        db_features = set(features).intersection(blast_features)
        if len(db_features) > 0:
            for protein_id, group in variants.groupby('protein_id'):
                _blast_protein_features(variants, group.index, db_features, db)


def _blast_protein_features(variants, indices, features, db):
    # Get blast
    protein_id, sequence = variants.loc[indices[0]][['protein_id', 'sequence']]
    blast = _get_blast(protein_id, sequence, db)

    # Compute features for each position
    for position, group in variants.loc[indices].groupby('position'):
        _blast_position_features(variants, group.index, features, db, blast)


def _blast_position_features(variants, indices, features, db, blast):
    # Get position column
    position = variants.loc[indices[0]].position
    try:
        aas = blast['aas'][:, position - 1]
        scores = blast['scores'][:, position - 1]
        evalue = blast['evalue'][:, position - 1]
    except:
        aas = np.array([], dtype=np.character)
        scores = np.array([], dtype=np.float)
        evalue = np.array([], dtype=np.float)

    aas_evalue = aas[evalue < _BLAST_EVALUE[db]]
    weights_evalue = scores[evalue < _BLAST_EVALUE[db]]
    aas_human = aas[blast['human']]
    weights_human = scores[blast['human']]
    aas_not_human = aas[np.logical_not(blast['human'])]
    weights_not_human = scores[np.logical_not(blast['human'])]

    for feature in features:
        feature_parts = feature.split('_')
        subset, aas_feature = feature_parts[2:4]

        if subset == 'all':
            weights = scores if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas, weights))
        elif subset == 'eva':
            weights = weights_evalue if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas_evalue, weights))
        elif subset == 'hum':
            weights = weights_human if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas_human, weights))
        elif subset == 'nhu':
            weights = weights_not_human if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas_not_human, weights))


def _aas_feature(variant, feature, aas, weights):
    if weights is None:
        weights = np.repeat(1., len(aas))

    def n_aln():
        return weights.sum()

    def wt_aas():
        return float(((aas == variant.wt.encode())*weights).sum())

    def mt_aas():
        return float(((aas == variant.mt.encode())*weights).sum())

    def difference_to_property_mean(property_dict):
        property_sum = 0.0
        property_n = 0.0
        for k, v in property_dict.items():
            n = (aas == k.encode()).sum()
            property_n += n
            property_sum += n*v
        return property_sum/property_n - property_dict[variant.mt]

    if feature == 'nal':
        return n_aln()
    elif feature == 'naa':
        return float(((aas != b'-')*weights).sum())
    elif feature == 'nwt':
        return wt_aas()
    elif feature == 'nmt':
        return mt_aas()
    elif feature == 'rwt':
        return wt_aas() / len(aas) if len(aas) > 0 else 0.0
    elif feature == 'rmt':
        return mt_aas() / len(aas) if len(aas) > 0 else 0.0
    elif feature == 'pwm':
        return _pssm(variant, wt_aas(), mt_aas())
    elif feature == 'dh':
        return difference_to_property_mean(_AA_Hkd)


def _pssm(variant, wt_x, mt_x):
    try:
        return (math.log(mt_x/_AA_Freq[variant.mt]) -
                math.log(wt_x/_AA_Freq[variant.wt]))
    except ValueError:
        return -10.


# Composition in percent for the complete Swissprot database
# (only Organism: Human)
_AA_Freq = {
    "A": 7.015, "C": 2.299, "E": 7.094, "D": 4.737, "G": 6.577, "F": 3.655,
    "I": 4.340, "H": 2.631, "K": 5.723, "M": 2.131, "L": 9.962, "N": 3.590,
    "Q": 4.764, "P": 6.306, "S": 8.318, "R": 5.645, "T": 5.356, "W": 1.220,
    "V": 5.973, "Y": 2.666,
    'U': sys.float_info.min, 'X': sys.float_info.min  # Invented
}


###################
# Kalign features #
###################

def _kalign_features(variants, features):
    logging.info('Computing kalign features for %d variants...' %
                 len(variants))
    for db, kalign_features in _KALIGN_FEATURES.items():
        db_features = set(features).intersection(kalign_features)
        if len(db_features) > 0:
            for protein_id, group in variants.groupby('protein_id'):
                _kalign_protein_features(
                    variants, group.index, db_features, db)


def _kalign_protein_features(variants, indices, features, db):
    # Get kalign
    protein_id, sequence = variants.loc[indices[0]][['protein_id', 'sequence']]
    kalign = _get_kalign(protein_id, sequence, db)

    # Compute features for each position
    for position, group in variants.loc[indices].groupby('position'):
        _kalign_position_features(variants, group.index, features, db, kalign)


def _kalign_position_features(variants, indices, features, db, kalign):
    # Get position column
    position = variants.loc[indices[0]].position
    aas = kalign['aligned_aas'][:, position - 1]

    aas_human = aas[kalign['human']]
    weights_human = kalign['similarity'][kalign['human']]
    aas_not_human = aas[np.logical_not(kalign['human'])]
    weights_not_human = kalign['similarity'][np.logical_not(kalign['human'])]

    for feature in features:
        feature_parts = feature.split('_')
        subset, aas_feature = feature_parts[2:4]

        if subset == 'all':
            weights = kalign['similarity'] if feature_parts[-1] == 'w' \
                else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas, weights))
        elif subset == 'hum':
            weights = weights_human if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas_human, weights))
        elif subset == 'nhu':
            weights = weights_not_human if feature_parts[-1] == 'w' else None
            for i in indices:
                variants.set_value(i, feature, _aas_feature(
                    variants.loc[i], aas_feature, aas_not_human, weights))


#############
# Get BLAST #
#############

def _get_blast_alignments(f):
    # zcat UniRef100_P13807.100.xml.gz
    #  | awk '/<Iteration>/{i++}i==2{print}'
    #  | grep 'Hit_def\|query-from\|query-to\|hseq\|qseq\|bit-score\|evalue'
    #  | sed 's/<Hit_def>/<Hit_def>!!!/g'
    #  | awk -F '[<>]' '//{print $3}'
    Alignment = collections.namedtuple('Alignment', 'title, hsps')
    Hsp = collections.namedtuple(
        'Hsp', 'query_start, query_end, sbjct, query, bits, expect')

    re_inside_tags = re.compile(b'>(.*)<')

    def get_field(s):
        return re_inside_tags.search(line).groups()[0]

    title = ''
    hsps = []
    for line in f:
        if line == b'</Iteration>\n':
            break

    for line in f:
        if line == b'<Hit>\n':
            for line in f:
                if line.startswith(b'  <Hit_def>'):
                    title = line.split(b'>')[1].split(b'<')[0].decode()
                if line == b'  <Hit_hsps>\n':
                    hsps = []
                    for line in f:
                        if line == b'    <Hsp>\n':
                            hsp = {}
                            for line in f:
                                if line.startswith(b'      <Hsp_query-from>'):
                                    hsp['query_start'] = int(get_field(line))
                                if line.startswith(b'      <Hsp_query-to>'):
                                    hsp['query_end'] = int(get_field(line))
                                if line.startswith(b'      <Hsp_hseq>'):
                                    hsp['sbjct'] = get_field(line)
                                if line.startswith(b'      <Hsp_qseq>'):
                                    hsp['query'] = get_field(line)
                                if line.startswith(b'      <Hsp_bit-score>'):
                                    hsp['bits'] = float(get_field(line))
                                if line.startswith(b'      <Hsp_evalue>'):
                                    hsp['expect'] = float(get_field(line))
                                if line == b'    </Hsp>\n':
                                    break
                            hsps.append(Hsp(**hsp))
                        elif line == b'  </Hit_hsps>\n':
                            break
                elif line == b'</Hit>\n':
                    break
            yield Alignment(title, hsps)


def _get_blast(protein_id, sequence, db):
    with io.BufferedReader(gzip.open(path_blast(protein_id, db), 'rb')) as f:
        aas = []
        scores = []
        evalue = []
        human = []

        for alignment in _get_blast_alignments(f):
            # New row
            new_aas = np.array([b'-']*len(sequence), dtype=np.character)
            new_scores = np.full(len(sequence), 0.0)
            new_evalue = np.full(len(sequence), 1.0)
            new_human = 'Homo sapiens' in alignment.title

            for hsp in alignment.hsps:
                start = hsp.query_start - 1
                end = hsp.query_end

                if (hsp.query == hsp.sbjct and
                        start == 0 and end == len(sequence)):
                    pass  # Exact match

                if (new_aas[start:end] != b'-').any():
                    # Append row
                    aas.append(new_aas)
                    scores.append(new_scores)
                    evalue.append(new_evalue)
                    human.append(new_human)

                    # New row
                    new_aas = np.array([b'-']*len(sequence),
                                       dtype=np.character)
                    new_scores = np.full(len(sequence), 0.0)
                    new_evalue = np.full(len(sequence), 1.0)
                    new_human = 'Homo sapiens' in alignment.title

                new_aas[start:end] = [
                    chr(s)
                    for q, s in zip(hsp.query, hsp.sbjct)
                    if q != ord(b'-')]
                new_scores[start:end] = hsp.bits  # Use bit score (normalized)
                new_evalue[start:end] = hsp.expect

            # Append row
            aas.append(new_aas)
            scores.append(new_scores)
            evalue.append(new_evalue)
            human.append(new_human)

    return {
        'aas': np.array(aas, dtype=np.character, order='F'),
        'scores': np.array(scores, order='F'),
        'evalue': np.array(evalue, order='F'),
        'human': np.array(human, dtype=np.bool),
    }


##############
# Get Kalign #
##############

def _get_kalign(protein_id, sequence, db):
    with io.BufferedReader(
            gzip.open(path_kalign(protein_id, db), 'rb')) as f:
        # Find protein in alignment
        self_name = uniref100(protein_id)
        aas = []
        names = []
        titles = []
        for title, seq in load_fasta_iter(f):
            titles.append(title)
            if '|' in title:
                name = title.split('|')[1].split()[0]
            else:
                name = title
            names.append(name)
            if self_name == name:
                # Protein sequence amino acid positions in alignment
                self_seq = seq
                seq_indices = np.where(seq != b'-')[0]
            aas.append(seq)

        if self_name not in names:
            raise KeyError('%s not found in %s.%s.afa.gz' % (
                self_name, self_name, db))

        similarity = np.array([
            float(((self_seq == row) & (row != b'-')).sum()) /
            len(sequence)
            for row in aas
        ])

        return {
            'aligned_aas': np.array(
                [seq[seq_indices] for seq in aas], dtype=np.character),
            'similarity': similarity,
            'names': names,
            'human': np.array([
                'Homo sapiens' in title for title in titles],
                dtype=np.bool)
        }


#########################
# Features distribution #
#########################

def features_distribution(variants, features=None):
    if features is None:
        features = [c for c in variants.columns if 'feature_' in c]

    sns.set_style("white")
    figures = []

    for feature in features:
        fig = plt.figure()

        plt.hist(
            variants[
                np.logical_not(variants.disease)][feature].dropna().values,
            label='Neutral', bins=20, alpha=0.5, color='g', normed=True,
            range=((variants[feature].min(), variants[feature].max())))
        plt.hist(variants[variants.disease][feature].dropna().values,
                 label='Disease', bins=20, alpha=0.5, color='r', normed=True,
                 range=((variants[feature].min(), variants[feature].max())))

        plt.legend(loc='upper right')
        plt.title(feature_name(feature), y=1.02)
        plt.locator_params(axis='y', nbins=6)

        figures.append((feature_name(feature), fig))

    return figures

