# -*- coding: utf-8 -*-

import csv
import io
import logging
import os
import re
import subprocess

from .utils import load_fasta_str

# Paths

_PROJECT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir)
_DATA_DIR = os.getenv('PMUT_DATA_DIR', os.path.join(_PROJECT_DIR, 'data'))
_VENDOR_DIR = os.path.join(_PROJECT_DIR, 'vendor')

_HUMAN_UNIREF100 = os.path.join(
    _VENDOR_DIR, 'Uniref100_human', 'UniRef100_human')
_BLASTDBCMD = os.path.join(
    _VENDOR_DIR, 'blastdbcmd')

##############################
# Blast, Fa and Kalign paths #
##############################


def _return_fname_if_exists(fname):
    if os.path.isfile(fname):
        return fname
    else:
        raise Exception('File not found: %s' % fname)


def path_fasta(protein_id, db):
    """Get path of fasta file, raise Exception if not found"""
    return _return_fname_if_exists(os.path.join(
        _DATA_DIR, 'fasta', uniref100(protein_id) + '.fasta'))


def _get_file(name, protein_id, db, extension):
    return _return_fname_if_exists(os.path.join(
        _DATA_DIR, '%s_%s' % (name, db),
        '%s.%s.%s' % (uniref100(protein_id), db, extension)))


def path_blast(protein_id, db):
    """Get path of blast file, raise Exception if not found"""
    return _get_file('blast', protein_id, db, 'xml.gz')


def path_fa(protein_id, db):
    """Get path of fa file, raise Exception if not found"""
    return _get_file('fa', protein_id, db, 'fa.gz')


def path_kalign(protein_id, db):
    """Get path of kalign file, raise Exception if not found"""
    return _get_file('kalign', protein_id, db, 'afa.gz')


##################################
# UniProt -> UniRef100, UniRef50 #
##################################

_UNIREF100_2_UNIPROT_FILE = os.path.join(
    _VENDOR_DIR, 'human_uniref100_2_uniprot.txt')

_UNIREF50_2_UNIPROT_FILE = os.path.join(
    _VENDOR_DIR, 'human_uniref50_2_uniprot.txt')  # NOTE: only human

_UNIPROT_2_UNIREF100 = {}
_UNIPROT_2_UNIREF50 = {}


def _load_uniref_to_uniprot(fname):
    with open(fname) as f:
        return {
            prot.strip(): line.split('\t')[0]
            for line in f.read().splitlines()
            for prot in line.split('\t')[1].split('; ')
        }


def uniref100(uniprot):
    """Return uniref100 cluster for uniprot if found,
    return uniprot otherwise"""
    global _UNIPROT_2_UNIREF100
    if not _UNIPROT_2_UNIREF100:
        _UNIPROT_2_UNIREF100 = _load_uniref_to_uniprot(
            _UNIREF100_2_UNIPROT_FILE)
    return _UNIPROT_2_UNIREF100.get(uniprot, uniprot)


def uniref50(uniprot):
    """Return uniref100 cluster for uniprot if found,
    return uniprot otherwise"""
    global _UNIPROT_2_UNIREF50
    if not _UNIPROT_2_UNIREF50:
        _UNIPROT_2_UNIREF50 = _load_uniref_to_uniprot(_UNIREF50_2_UNIPROT_FILE)
    return _UNIPROT_2_UNIREF50.get(uniprot, uniprot)

######################
# Network parameters #
######################

_NETWORK_PARAM_FILE = os.path.join(
    _VENDOR_DIR, 'topological_characteristics_binary_interaction_data.tsv')


def get_network_parameters():
    with open(_NETWORK_PARAM_FILE) as f:
        reader = csv.reader(f, delimiter='\t')
        header = next(reader)
        return {
            row[0]: [float(x) for x in row[1:]]
            for row in csv.reader(f, delimiter='\t')
        }, {
            'feature_top_' + h: i for i, h in enumerate(header[1:])
        }


######################
# Blast DB sequences #
######################

def get_protein_sequences(protein_ids):
    uniref100_ids = [uniref100(protein_id) for protein_id in protein_ids]
    p = subprocess.Popen([
        _BLASTDBCMD, '-db', _HUMAN_UNIREF100, '-dbtype', 'prot',
        '-entry_batch', '/proc/self/fd/0'],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE)
    out, _ = p.communicate(('\n'.join(uniref100_ids)).encode())
    fasta = {
        title.split('|')[1].split()[0]: sequence
        for title, sequence in load_fasta_str(io.BytesIO(out)).items()
    }
    missing = set(uniref100_ids).difference(fasta.keys())
    if missing:
        logging.warn('Sequence not found for %d ids: %s', len(missing),
                     str(list(missing)))
    return {
        protein_id: fasta[uniref100(protein_id)]
        for protein_id in protein_ids
        if uniref100(protein_id) in fasta
    }
