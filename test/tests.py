#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys; sys.path.insert(1, '..')
import os; TEST_DIR = os.path.dirname(os.path.abspath(__file__))
import os; os.environ['PMUT_DATA_DIR'] = os.path.join(TEST_DIR, '../test/data')
import matplotlib; matplotlib.use('Agg')

import io
import itertools
import logging
import os
import unittest
import sys

import numpy as np
import pandas as pd

from pymut import data
from pymut.amino_acids import (
    AMINO_ACIDS, AA3, AA1, AA_NAME, AA_DISTANCE, aa1, aa3, aa_distance,
    aa_name)
from pymut.data import (
    _PROJECT_DIR, _DATA_DIR, _VENDOR_DIR, _HUMAN_UNIREF100, _BLASTDBCMD,
    path_fasta, path_blast, path_fa, path_kalign, uniref100, uniref50,
    _UNIREF100_2_UNIPROT_FILE, _UNIREF50_2_UNIPROT_FILE, _NETWORK_PARAM_FILE,
    get_protein_sequences
)
from pymut.features import (
    FEATURES, PMUT_FEATURES, feature_name, compute_features,
    _DBS, _BLAST_FEATURES, _KALIGN_FEATURES, _FEATURES,
    _get_blast, _get_kalign, _basic_features,
    _topology_features, _blast_features, _kalign_features,
    features_distribution)
from pymut.io import read_humsavar_txt
from pymut.pmut import get_sequences
from pymut.utils import (
    variant_str, load_fasta_str, load_fasta_iter, all_variants, parse_variant)

# Disable logging
logging.disable(logging.INFO)

# Test constants

P13807_SEQUENCE = (
    'MPLNRTLSMSSLPGLEDWEDEFDLENAVLFEVAWEVANKVGGIYTVLQTKAKVTGDEWGDNYFL' +
    'VGPYTEQGVRTQVELLEAPTPALKRTLDSMNSKGCKVYFGRWLIEGGPLVVLLDVGASAWALER' +
    'WKGELWDTCNIGVPWYDREANDAVLFGFLTTWFLGEFLAQSEEKPHVVAHFHEWLAGVGLCLCR' +
    'ARRLPVATIFTTHATLLGRYLCAGAVDFYNNLENFNVDKEAGERQIYHRYCMERAAAHCAHVFT' +
    'TVSQITAIEAQHLLKRKPDIVTPNGLNVKKFSAMHEFQNLHAQSKARIQEFVRGHFYGHLDFNL' +
    'DKTLYFFIAGRYEFSNKGADVFLEALARLNYLLRVNGSEQTVVAFFIMPARTNNFNVETLKGQA' +
    'VRKQLWDTANTVKEKFGRKLYESLLVGSLPDMNKMLDKEDFTMMKRAIFATQRQSFPPVCTHNM' +
    'LDDSSDPILTTIRRIGLFNSSADRVKVIFHPEFLSSTSPLLPVDYEEFVRGCHLGVFPSYYEPW' +
    'GYTPAECTVMGIPSISTNLSGFGCFMEEHIADPSAYGIYILDRRFRSLDDSCSQLTSFLYSFCQ' +
    'QSRRQRIIQRNRTERLSDLLDWKYLGRYYMSARHMALSKAFPEHFTYEPNEADAAQGYRYPRPA' +
    'SVPPSPSLSRHSSPHQSEDEEDPRNGPLEEDGERYDEDEEAAKDRRNIRAPEWPRRASCTSSTS' +
    'GSKRNSVDTATSSSLSTPSEPLSPTSSLGEERN')
variants_P13807 = all_variants('P13807', P13807_SEQUENCE)

L8EA00_SEQUENCE = 'MMEKSGGCMPTYMEMGRVKSLMKMKRHSSASQRHTSVTPSPIVGRKILHHLRKLFKEP'
variants_L8EA00 = all_variants('L8EA00', L8EA00_SEQUENCE)


###############
#   pmut.py   #
###############

class PmutTests(unittest.TestCase):
    def testGetSequences(self):
        # Prepare variants
        wrong_variant = pd.DataFrame(
            [{'protein_id': 'P13807','wt': 'A', 'mt': 'A', 'position': 1}])
        variants = pd.concat([
            wrong_variant,
            variants_P13807.sample(3, random_state=42),
            variants_L8EA00.sample(3, random_state=42)])

        # Warning message for variant with wrong WT
        with self.assertLogs(level='WARNING'):
            variants = get_sequences(variants)

        # There are 6 variants, the wrong one was discarded
        self.assertEqual(len(variants), 6)

        # Added sequence column to variants
        self.assertIn('sequence', variants.columns)

        # Check that sequence strings *are* the same
        self.assertIs(variants.iloc[0].sequence, variants.iloc[1].sequence)
        self.assertIs(variants.iloc[1].sequence, variants.iloc[2].sequence)
        self.assertIs(variants.iloc[3].sequence, variants.iloc[4].sequence)
        self.assertIs(variants.iloc[4].sequence, variants.iloc[5].sequence)


###################
#   features.py   #
###################

class FeaturesTests(unittest.TestCase):
    def testFeatureConstants(self):
        # All feature names are different
        self.assertEqual(len(FEATURES), len(set(FEATURES)))

        # Blast and kalign features defined for each DB
        for db in _DBS:
            self.assertGreater(len(_BLAST_FEATURES[db]), 0)
            self.assertGreater(len(_KALIGN_FEATURES[db]), 0)

        # Pmut features are in FEATURES
        self.assertLessEqual(set(PMUT_FEATURES), set(FEATURES))

    def testFeatureName(self):
        # All feature names are disjoint
        self.assertEqual(
            len(FEATURES),
            len(set(feature_name(feature) for feature in FEATURES)))

    # General features computation

    def testComputePMutFeatures(self):
        variants = variants_L8EA00.sample(2, random_state=42)
        compute_features(variants, PMUT_FEATURES)
        # All features are set
        self.assertEqual(
            set([c for c in variants.columns if c.startswith('feature_')]),
            set(PMUT_FEATURES))

    def testComputeAllFeatures(self):
        variants = variants_L8EA00.sample(2, random_state=42)
        compute_features(variants)
        # All features are set
        self.assertEqual(
            set([c for c in variants.columns if c.startswith('feature_')]),
            set(FEATURES))

    # Basic features

    def testBasicFeatures(self):
        variants = pd.DataFrame([{
            'protein_id': 'P13807',
            'wt': 'M',
            'mt': 'A',
            'position': 1,
            'sequence': 'MPLNRTLSMSSLPGLEDWEDEFDLENAVLFE...'
        }])

        _basic_features(variants, _FEATURES['basic'])

        variant = variants.iloc[0]
        # Generated from example run
        self.assertAlmostEqual(variant.feature_pos, 1)
        self.assertAlmostEqual(variant.feature_vol_d, 79.)
        self.assertAlmostEqual(variant.feature_vol_r, 0.4619883040935672)
        self.assertAlmostEqual(variant.feature_hww_d, 0.4)
        self.assertAlmostEqual(variant.feature_hww_r, 1.7391304347826086)
        self.assertAlmostEqual(variant.feature_hwo_r, 0.75)
        self.assertAlmostEqual(variant.feature_hwo_d, 1.26)
        self.assertAlmostEqual(variant.feature_hkd, 0.020000000000000108)
        self.assertAlmostEqual(variant.feature_blosum62, -1)
        self.assertAlmostEqual(variant.feature_blosum50, -1)
        self.assertAlmostEqual(variant.feature_blosum80, -1)
        self.assertAlmostEqual(variant.feature_miyata, -1.17)
        self.assertAlmostEqual(variant.feature_pam60, -3)

    # Topology features

    def testTopologyFeatures(self):
        variants = pd.DataFrame([{
            'protein_id': 'P13807',
        }])

        P13807_TOPOLOGY = {
            'degree': 13,
            'degree_centrality': 0.00101261878797,
            'betweenness': 0.000145575281033,
            'closeness': 0.289818521285,
            'eigenvector': 0.0086213569834,
            'cc': 0.025641025641
        }

        # Compute only one feature
        _topology_features(variants, ['feature_top_cc'])
        self.assertEqual(
            list(variants.columns), ['protein_id', 'feature_top_cc'])

        # Compute all features
        _topology_features(variants, _FEATURES['topology'])
        variant = variants.iloc[0]
        for k, v in P13807_TOPOLOGY.items():
            self.assertAlmostEqual(variant['feature_top_' + k], v)

    # Blast features

    def _check_features(self, variants, expected_features):
        for (i, variant), expected_features in zip(
                variants.iterrows(), expected_features):
            for k, v in expected_features.items():
                # print(k, '\t', variant['feature_' + k], v)
                if 'pwm' in k and variant['feature_' + k] == -10.:
                    # Ignore because PWM computation changed
                    continue
                self.assertAlmostEqual(variant['feature_' + k], v)

    def testBlastFeatures(self):
        # Long test
        variants = variants_P13807.sample(10, random_state=42)
        _blast_features(variants, _FEATURES['blast'])
        # expected_features = [
        #     {'b1_eva_nmt_w': 24408.148, 'b1_eva_pwm_w': -1.3018590148442382, 'b1_hum_rwt_w': 364.8815833333333, 'b1_all_rmt_w': 15.074872235872236, 'b1_all_nwt_w': 150128.22, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 43.0, 'b1_eva_nwt': 159.0, 'b1_all_rwt_w': 92.21635135135135, 'b1_all_rwt': 0.0988943488943489, 'b1_all_naa': 1225.0, 'b1_all_nmt': 44.0, 'b1_eva_rmt_w': 19.876342019543976, 'b1_hum_pwm_w': -714.4821201991211, 'b1_hum_nwt': 5.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1191.0, 'b1_eva_rwt_w': 121.8654706840391, 'b1_eva_nal': 1228, 'b1_hum_nwt_w': 4378.579, 'b1_all_pwm_w': -1.2995796531365489, 'b1_eva_pwm': -0.7961755456257464, 'b1_eva_rmt': 0.03501628664495114, 'b1_hum_rwt': 0.4166666666666667, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 149650.798, 'b1_all_nwt': 161.0, 'b1_hum_nal': 12, 'b1_all_pwm': -0.7856861901652794, 'b1_eva_rwt': 0.12947882736156352, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 24541.892, 'b1_all_rmt': 0.02702702702702703, 'b1_hum_pwm': -707.7070785900471},
        #     {'b1_eva_nmt_w': 2368.143, 'b1_eva_pwm_w': -5.044516636515181, 'b1_hum_rwt_w': 444.30358333333334, 'b1_all_rmt_w': 1.4546332923832923, 'b1_all_nwt_w': 463761.76700000005, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 4.0, 'b1_eva_nwt': 550.0, 'b1_all_rwt_w': 284.8659502457003, 'b1_all_rwt': 0.3396805896805897, 'b1_all_naa': 1462.0, 'b1_all_nmt': 4.0, 'b1_eva_rmt_w': 1.672417372881356, 'b1_hum_pwm_w': -715.1905839433324, 'b1_hum_nwt': 6.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1411.0, 'b1_eva_rwt_w': 327.21977330508474, 'b1_eva_nal': 1416, 'b1_hum_nwt_w': 5331.643, 'b1_all_pwm_w': -5.045419593664133, 'b1_eva_pwm': -4.691778627298072, 'b1_eva_rmt': 0.002824858757062147, 'b1_hum_rwt': 0.5, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 463343.199, 'b1_all_nwt': 553.0, 'b1_hum_nal': 12, 'b1_all_pwm': -4.69721835059389, 'b1_eva_rwt': 0.3884180790960452, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 2368.143, 'b1_all_rmt': 0.002457002457002457, 'b1_hum_pwm': -708.400928687742},
        #     {'b1_eva_nmt_w': 53662.688, 'b1_eva_pwm_w': -0.28789642664668413, 'b1_hum_rwt_w': 364.8815833333333, 'b1_all_rmt_w': 33.516486486486485, 'b1_all_nwt_w': 150128.22, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 74.0, 'b1_eva_nwt': 159.0, 'b1_all_rwt_w': 92.21635135135135, 'b1_all_rwt': 0.0988943488943489, 'b1_all_naa': 1225.0, 'b1_all_nmt': 79.0, 'b1_eva_rmt_w': 43.69925732899023, 'b1_hum_pwm_w': -714.4821201991211, 'b1_hum_nwt': 5.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1191.0, 'b1_eva_rwt_w': 121.8654706840391, 'b1_eva_nal': 1228, 'b1_hum_nwt_w': 4378.579, 'b1_all_pwm_w': -0.27440980652660407, 'b1_eva_pwm': -0.027148905858024097, 'b1_eva_rmt': 0.06026058631921824, 'b1_hum_rwt': 0.4166666666666667, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 149650.798, 'b1_all_nwt': 161.0, 'b1_hum_nal': 12, 'b1_all_pwm': 0.025733690640596052, 'b1_eva_rwt': 0.12947882736156352, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 54564.840000000004, 'b1_all_rmt': 0.048525798525798525, 'b1_hum_pwm': -707.7070785900471},
        #     {'b1_eva_nmt_w': 2649.337, 'b1_eva_pwm_w': -5.558521874534306, 'b1_hum_rwt_w': 695.47375, 'b1_all_rmt_w': 1.6273568796068796, 'b1_all_nwt_w': 735729.514, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 4.0, 'b1_eva_nwt': 953.0, 'b1_all_rwt_w': 451.9223058968059, 'b1_all_rwt': 0.6136363636363636, 'b1_all_naa': 1470.0, 'b1_all_nmt': 4.0, 'b1_eva_rmt_w': 1.8683617771509167, 'b1_hum_pwm_w': -715.6386691363219, 'b1_hum_nwt': 9.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1408.0, 'b1_eva_rwt_w': 512.8908131170664, 'b1_eva_nal': 1418, 'b1_hum_nwt_w': 8345.685, 'b1_all_pwm_w': -5.570074006864011, 'b1_eva_pwm': -5.416841426385767, 'b1_eva_rmt': 0.0028208744710860366, 'b1_hum_rwt': 0.75, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 727279.1730000001, 'b1_all_nwt': 999.0, 'b1_hum_nal': 12, 'b1_all_pwm': -5.4639813013801195, 'b1_eva_rwt': 0.6720733427362482, 'b1_hum_naa': 9.0, 'b1_all_nmt_w': 2649.337, 'b1_all_rmt': 0.002457002457002457, 'b1_hum_pwm': -708.8063937958501},
        #     {'b1_eva_nmt_w': 13478.337, 'b1_eva_pwm_w': -2.0786929303628643, 'b1_hum_rwt_w': 466.51675, 'b1_all_rmt_w': 8.38157800982801, 'b1_all_nwt_w': 288104.216, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 17.0, 'b1_eva_nwt': 324.0, 'b1_all_rwt_w': 176.96819164619166, 'b1_all_rwt': 0.20393120393120392, 'b1_all_naa': 886.0, 'b1_all_nmt': 18.0, 'b1_eva_rmt_w': 15.368685290763967, 'b1_hum_pwm_w': -715.0673697510241, 'b1_hum_nwt': 6.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 867.0, 'b1_eva_rwt_w': 326.91332725199544, 'b1_eva_nal': 877, 'b1_hum_nwt_w': 5598.201, 'b1_all_pwm_w': -2.0712636701872817, 'b1_eva_pwm': -1.9688600371412, 'b1_eva_rmt': 0.019384264538198404, 'b1_hum_rwt': 0.5, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 286702.988, 'b1_all_nwt': 332.0, 'b1_hum_nal': 12, 'b1_all_pwm': -1.936093076425411, 'b1_eva_rwt': 0.3694412770809578, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 13645.208999999999, 'b1_all_rmt': 0.011056511056511056, 'b1_hum_pwm': -708.2289286451407},
        #     {'b1_eva_nmt_w': 2493.4929999999995, 'b1_eva_pwm_w': -5.998217647056036, 'b1_hum_rwt_w': 717.822, 'b1_all_rmt_w': 1.531629606879607, 'b1_all_nwt_w': 906463.7596000001, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 6.0, 'b1_eva_nwt': 1180.0, 'b1_all_rwt_w': 556.7959211302212, 'b1_all_rwt': 0.7401719901719902, 'b1_all_naa': 1225.0, 'b1_all_nmt': 6.0, 'b1_eva_rmt_w': 2.0305317589576544, 'b1_hum_pwm_w': -715.6160452483789, 'b1_hum_nwt': 10.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1191.0, 'b1_eva_rwt_w': 735.0703460912052, 'b1_eva_nal': 1228, 'b1_hum_nwt_w': 8613.864, 'b1_all_pwm_w': -6.002415664747804, 'b1_eva_pwm': -5.38805940066321, 'b1_eva_rmt': 0.004885993485342019, 'b1_hum_rwt': 0.8333333333333334, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 902666.385, 'b1_all_nwt': 1205.0, 'b1_hum_nal': 12, 'b1_all_pwm': -5.4090245291282555, 'b1_eva_rwt': 0.9609120521172638, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 2493.493, 'b1_all_rmt': 0.0036855036855036856, 'b1_hum_pwm': -708.8575020641352},
        #     {'b1_eva_nmt_w': 0.0, 'b1_eva_pwm_w': -720.198381445033, 'b1_hum_rwt_w': 638.4, 'b1_all_rmt_w': 0.0, 'b1_all_nwt_w': 882617.3955999999, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 0.0, 'b1_eva_nwt': 1157.0, 'b1_all_rwt_w': 542.1482773955773, 'b1_all_rwt': 0.7297297297297297, 'b1_all_naa': 1203.0, 'b1_all_nmt': 0.0, 'b1_eva_rmt_w': 0.0, 'b1_hum_pwm_w': -715.4567115138684, 'b1_hum_nwt': 9.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1168.0, 'b1_eva_rwt_w': 715.0836425081433, 'b1_eva_nal': 1228, 'b1_hum_nwt_w': 7660.799999999999, 'b1_all_pwm_w': -720.2034869030629, 'b1_eva_pwm': -713.5664255451385, 'b1_eva_rmt': 0.0, 'b1_hum_rwt': 0.75, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 878122.713, 'b1_all_nwt': 1188.0, 'b1_hum_nal': 12, 'b1_all_pwm': -713.5928663178674, 'b1_eva_rwt': 0.9421824104234527, 'b1_hum_naa': 9.0, 'b1_all_nmt_w': 0.0, 'b1_all_rmt': 0.0, 'b1_hum_pwm': -708.710064395281},
        #     {'b1_eva_nmt_w': 0.0, 'b1_eva_pwm_w': -720.3284642013953, 'b1_hum_rwt_w': 740.0351666666666, 'b1_all_rmt_w': 0.0, 'b1_all_nwt_w': 661674.6211999999, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 0.0, 'b1_eva_nwt': 827.0, 'b1_all_rwt_w': 406.4340425061425, 'b1_all_rwt': 0.5141277641277642, 'b1_all_naa': 1397.0, 'b1_all_nmt': 0.0, 'b1_eva_rmt_w': 0.0, 'b1_hum_pwm_w': -716.0201485415314, 'b1_hum_nwt': 10.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1368.0, 'b1_eva_rwt_w': 479.6151438953489, 'b1_eva_nal': 1376, 'b1_hum_nwt_w': 8880.421999999999, 'b1_all_pwm_w': -720.3310733894796, 'b1_eva_pwm': -713.6463488791754, 'b1_eva_rmt': 0.0, 'b1_hum_rwt': 0.8333333333333334, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 659950.4380000001, 'b1_all_nwt': 837.0, 'b1_hum_nal': 12, 'b1_all_pwm': -713.6583682546412, 'b1_eva_rwt': 0.6010174418604651, 'b1_hum_naa': 10.0, 'b1_all_nmt_w': 0.0, 'b1_all_rmt': 0.0, 'b1_hum_pwm': -709.2311292771458},
        #     {'b1_eva_nmt_w': 691.513, 'b1_eva_pwm_w': -7.666972597751078, 'b1_hum_rwt_w': 695.47375, 'b1_all_rmt_w': 0.5050221130221131, 'b1_all_nwt_w': 814879.3879999999, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 1.0, 'b1_eva_nwt': 1052.0, 'b1_all_rwt_w': 500.54016461916456, 'b1_all_rwt': 0.6744471744471745, 'b1_all_naa': 1477.0, 'b1_all_nmt': 2.0, 'b1_eva_rmt_w': 0.48766784203102964, 'b1_hum_pwm_w': -716.1477662475719, 'b1_hum_nwt': 9.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1415.0, 'b1_eva_rwt_w': 568.7411043723555, 'b1_eva_nal': 1418, 'b1_hum_nwt_w': 8345.685, 'b1_all_pwm_w': -7.504267418846715, 'b1_eva_pwm': -7.563874905116682, 'b1_eva_rmt': 0.0007052186177715092, 'b1_hum_rwt': 0.75, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 806474.886, 'b1_all_nwt': 1098.0, 'b1_hum_nal': 12, 'b1_all_pwm': -6.913524953328556, 'b1_eva_rwt': 0.7418899858956276, 'b1_hum_naa': 9.0, 'b1_all_nmt_w': 822.176, 'b1_all_rmt': 0.0012285012285012285, 'b1_hum_pwm': -709.3154909071001},
        #     {'b1_eva_nmt_w': 12728.067, 'b1_eva_pwm_w': -4.222681791451878, 'b1_hum_rwt_w': 745.1778333333332, 'b1_all_rmt_w': 8.023934889434889, 'b1_all_nwt_w': 733657.8405999999, 'b1_hum_nmt': 0.0, 'b1_eva_nmt': 23.0, 'b1_eva_nwt': 869.0, 'b1_all_rwt_w': 450.6497792383292, 'b1_all_rwt': 0.5571253071253072, 'b1_all_naa': 1289.0, 'b1_all_nmt': 25.0, 'b1_eva_rmt_w': 10.248041062801931, 'b1_hum_pwm_w': -715.939544050466, 'b1_hum_nwt': 11.0, 'b1_all_nal': 1628, 'b1_hum_nmt_w': 0.0, 'b1_hum_rmt': 0.0, 'b1_eva_naa': 1239.0, 'b1_eva_rwt_w': 586.6346296296296, 'b1_eva_nal': 1242, 'b1_hum_nwt_w': 8942.133999999998, 'b1_all_pwm_w': -4.20362773263891, 'b1_eva_pwm': -3.8072150829962528, 'b1_eva_rmt': 0.018518518518518517, 'b1_hum_rwt': 0.9166666666666666, 'b1_hum_rmt_w': 0.0, 'b1_eva_nwt_w': 728600.21, 'b1_all_nwt': 907.0, 'b1_hum_nal': 12, 'b1_all_pwm': -3.7666327989069464, 'b1_eva_rwt': 0.6996779388083736, 'b1_hum_naa': 11.0, 'b1_all_nmt_w': 13062.966, 'b1_all_rmt': 0.015356265356265357, 'b1_hum_pwm': -709.2389097811208},
        # ]
        # self._check_features(variants, expected_features)

        # Short test
        variants = variants_L8EA00.sample(2, random_state=42)
        _blast_features(variants, _FEATURES['blast'])
        expected_features = [
            {'b9_all_nwt': 1.000000, 'b1_all_nwt': 1.000000, 'b9_hum_rmt': 0.000000, 'b1_hum_rwt_w': 94.789800, 'b9_eva_nmt': 0.000000, 'b9_all_rwt': 1.000000, 'b9_eva_rwt': 0.000000, 'b9_hum_rwt_w': 94.789800, 'b9_hum_nal_w': 94.789800, 'b9_eva_nmt_w': 0.000000, 'b9_hum_nwt': 1.000000, 'b9_eva_rmt': 0.000000, 'b1_all_nal_w': 94.789800, 'b9_eva_nwt_w': 0.000000, 'b1_all_rwt': 1.000000, 'b9_all_nmt': 0.000000, 'b1_eva_nwt': 0.000000, 'b1_hum_rmt': 0.000000, 'b9_all_rwt_w': 94.789800, 'b1_eva_nmt_w': 0.000000, 'b9_eva_pwm': -10.000000, 'b9_all_nal_w': 94.789800, 'b9_eva_pwm_w': -10.000000, 'b1_hum_nal': 1.000000, 'b9_all_nmt_w': 0.000000, 'b9_hum_nwt_w': 94.789800, 'b1_all_rmt': 0.000000, 'b9_all_rmt': 0.000000, 'b9_hum_naa_w': 94.789800, 'b9_hum_pwm_w': -10.000000, 'b1_all_naa': 1.000000, 'b1_eva_pwm_w': -10.000000, 'b1_eva_nwt_w': 0.000000, 'b1_eva_rwt_w': 0.000000, 'b1_hum_nmt': 0.000000, 'b9_eva_rmt_w': 0.000000, 'b1_hum_nmt_w': 0.000000, 'b1_hum_naa': 1.000000, 'b9_hum_pwm': -10.000000, 'b1_eva_naa': 0.000000, 'b1_eva_naa_w': 0.000000, 'b9_all_nwt_w': 94.789800, 'b1_hum_nwt_w': 94.789800, 'b1_eva_rmt_w': 0.000000, 'b1_all_nal': 1.000000, 'b9_eva_naa': 0.000000, 'b9_all_nal': 1.000000, 'b9_hum_nal': 1.000000, 'b9_hum_nmt_w': 0.000000, 'b9_eva_nal': 0.000000, 'b1_all_nmt_w': 0.000000, 'b1_eva_nmt': 0.000000, 'b1_eva_rmt': 0.000000, 'b9_eva_nal_w': 0.000000, 'b1_eva_rwt': 0.000000, 'b9_all_rmt_w': 0.000000, 'b1_all_rmt_w': 0.000000, 'b1_all_nmt': 0.000000, 'b9_eva_naa_w': 0.000000, 'b1_eva_pwm': -10.000000, 'b9_hum_naa': 1.000000, 'b9_hum_rmt_w': 0.000000, 'b9_hum_nmt': 0.000000, 'b1_hum_nal_w': 94.789800, 'b9_all_pwm': -10.000000, 'b1_eva_nal_w': 0.000000, 'b9_hum_rwt': 1.000000, 'b9_all_naa_w': 94.789800, 'b1_hum_pwm': -10.000000, 'b1_eva_nal': 0.000000, 'b1_hum_naa_w': 94.789800, 'b1_all_naa_w': 94.789800, 'b9_all_pwm_w': -10.000000, 'b1_all_rwt_w': 94.789800, 'b9_all_naa': 1.000000, 'b1_hum_pwm_w': -10.000000, 'b1_all_pwm_w': -10.000000, 'b1_hum_rmt_w': 0.000000, 'b1_all_pwm': -10.000000, 'b1_hum_nwt': 1.000000, 'b9_eva_nwt': 0.000000, 'b9_eva_rwt_w': 0.000000, 'b1_hum_rwt': 1.000000, 'b1_all_nwt_w': 94.789800},
            {'b9_all_nwt': 1.000000, 'b1_all_nwt': 1.000000, 'b9_hum_rmt': 0.000000, 'b1_hum_rwt_w': 94.789800, 'b9_eva_nmt': 0.000000, 'b9_all_rwt': 1.000000, 'b9_eva_rwt': 0.000000, 'b9_hum_rwt_w': 94.789800, 'b9_hum_nal_w': 94.789800, 'b9_eva_nmt_w': 0.000000, 'b9_hum_nwt': 1.000000, 'b9_eva_rmt': 0.000000, 'b1_all_nal_w': 94.789800, 'b9_eva_nwt_w': 0.000000, 'b1_all_rwt': 1.000000, 'b9_all_nmt': 0.000000, 'b1_eva_nwt': 0.000000, 'b1_hum_rmt': 0.000000, 'b9_all_rwt_w': 94.789800, 'b1_eva_nmt_w': 0.000000, 'b9_eva_pwm': -10.000000, 'b9_all_nal_w': 94.789800, 'b9_eva_pwm_w': -10.000000, 'b1_hum_nal': 1.000000, 'b9_all_nmt_w': 0.000000, 'b9_hum_nwt_w': 94.789800, 'b1_all_rmt': 0.000000, 'b9_all_rmt': 0.000000, 'b9_hum_naa_w': 94.789800, 'b9_hum_pwm_w': -10.000000, 'b1_all_naa': 1.000000, 'b1_eva_pwm_w': -10.000000, 'b1_eva_nwt_w': 0.000000, 'b1_eva_rwt_w': 0.000000, 'b1_hum_nmt': 0.000000, 'b9_eva_rmt_w': 0.000000, 'b1_hum_nmt_w': 0.000000, 'b1_hum_naa': 1.000000, 'b9_hum_pwm': -10.000000, 'b1_eva_naa': 0.000000, 'b1_eva_naa_w': 0.000000, 'b9_all_nwt_w': 94.789800, 'b1_hum_nwt_w': 94.789800, 'b1_eva_rmt_w': 0.000000, 'b1_all_nal': 1.000000, 'b9_eva_naa': 0.000000, 'b9_all_nal': 1.000000, 'b9_hum_nal': 1.000000, 'b9_hum_nmt_w': 0.000000, 'b9_eva_nal': 0.000000, 'b1_all_nmt_w': 0.000000, 'b1_eva_nmt': 0.000000, 'b1_eva_rmt': 0.000000, 'b9_eva_nal_w': 0.000000, 'b1_eva_rwt': 0.000000, 'b9_all_rmt_w': 0.000000, 'b1_all_rmt_w': 0.000000, 'b1_all_nmt': 0.000000, 'b9_eva_naa_w': 0.000000, 'b1_eva_pwm': -10.000000, 'b9_hum_naa': 1.000000, 'b9_hum_rmt_w': 0.000000, 'b9_hum_nmt': 0.000000, 'b1_hum_nal_w': 94.789800, 'b9_all_pwm': -10.000000, 'b1_eva_nal_w': 0.000000, 'b9_hum_rwt': 1.000000, 'b9_all_naa_w': 94.789800, 'b1_hum_pwm': -10.000000, 'b1_eva_nal': 0.000000, 'b1_hum_naa_w': 94.789800, 'b1_all_naa_w': 94.789800, 'b9_all_pwm_w': -10.000000, 'b1_all_rwt_w': 94.789800, 'b9_all_naa': 1.000000, 'b1_hum_pwm_w': -10.000000, 'b1_all_pwm_w': -10.000000, 'b1_hum_rmt_w': 0.000000, 'b1_all_pwm': -10.000000, 'b1_hum_nwt': 1.000000, 'b9_eva_nwt': 0.000000, 'b9_eva_rwt_w': 0.000000, 'b1_hum_rwt': 1.000000, 'b1_all_nwt_w': 94.789800},]
        self._check_features(variants, expected_features)

    # Kalign features

    def testKalignFeatures(self):
        # Long Test
        # variants = variants_P13807.sample(10, random_state=42)
        # _kalign_features(variants, _FEATURES['kalign'])
        # expected_features = [
        #     {'k9_all_nwt': 29.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -707.7070785900471, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.006035946353204284, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.4166666666666667, 'k9_hum_pwm': -706.097640677613, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -0.06701824305985082, 'k1_all_naa': 1181.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.026381909547738693, 'k1_hum_rwt_w': 0.331524197195839, 'k9_all_rwt': 0.040446304044630406, 'k1_all_pwm': -0.8006578480652462, 'k1_all_nwt': 156.0, 'k1_all_nmt_w': 9.60922659430122, 'k9_all_pwm': 0.7551506235586731, 'k9_all_rmt_w': 0.011316562868426978, 'k9_hum_nwt_w': 0.24151967435549526, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -707.4784928484302, 'k9_hum_pwm_w': -704.6768363357157, 'k9_all_naa': 511.0, 'k1_all_nwt_w': 126.50474898236091, 'k9_hum_nal': 3, 'k9_all_nmt_w': 8.113975576662144, 'k9_all_rwt_w': 0.020182465383239753, 'k1_all_pwm_w': -2.0660275674420174, 'k1_hum_nwt': 5.0, 'k1_all_rwt': 0.09798994974874371, 'k9_all_rmt': 0.05160390516039052, 'k9_all_nmt': 37.0, 'k1_hum_nwt_w': 3.978290366350068, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 42.0, 'k9_hum_rwt_w': 0.08050655811849843, 'k9_all_nwt_w': 14.470827679782904, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.07946278202409605},
        #     {'k9_all_nwt': 214.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -708.400928687742, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0005795599435440431, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.5, 'k9_hum_pwm': -706.6091692185139, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -4.575868564674781, 'k1_all_naa': 1427.0, 'k1_hum_naa': 10.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.0018844221105527637, 'k1_hum_rwt_w': 0.40750791497060157, 'k9_all_rwt': 0.298465829846583, 'k1_all_pwm': -4.972161397268241, 'k1_all_nwt': 546.0, 'k1_all_nmt_w': 0.9226594301221166, 'k9_all_pwm': -4.035518436545187, 'k9_all_rmt_w': 0.0011865359395491163, 'k9_hum_nwt_w': 0.24151967435549526, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.1963809450529, 'k9_hum_pwm_w': -705.1883648766167, 'k9_all_naa': 575.0, 'k1_all_nwt_w': 343.20624151967434, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.8507462686567164, 'k9_all_rwt_w': 0.14529293433933416, 'k1_all_pwm_w': -5.686981357905207, 'k1_hum_nwt': 6.0, 'k1_all_rwt': 0.34296482412060303, 'k9_all_rmt': 0.0041841004184100415, 'k9_all_nmt': 3.0, 'k1_hum_nwt_w': 4.890094979647219, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 3.0, 'k9_hum_rwt_w': 0.08050655811849843, 'k9_all_nwt_w': 104.17503392130259, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.21558180999979543},
        #     {'k9_all_nwt': 29.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -707.7070785900471, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.021397694033259924, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.4166666666666667, 'k9_hum_pwm': -706.097640677613, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': 1.0331478743106584, 'k1_all_naa': 1181.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.04836683417085427, 'k1_hum_rwt_w': 0.331524197195839, 'k9_all_rwt': 0.040446304044630406, 'k1_all_pwm': 0.03163961776218427, 'k1_all_nwt': 156.0, 'k1_all_nmt_w': 34.0651289009498, 'k9_all_pwm': 1.2205419748816222, 'k9_all_rmt_w': 0.02712001044605803, 'k9_hum_nwt_w': 0.24151967435549526, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -707.4784928484302, 'k9_hum_pwm_w': -704.6768363357157, 'k9_all_naa': 511.0, 'k1_all_nwt_w': 126.50474898236091, 'k9_hum_nal': 3, 'k9_all_nmt_w': 19.445047489823608, 'k9_all_rwt_w': 0.020182465383239753, 'k1_all_pwm_w': -0.5743153974785973, 'k1_hum_nwt': 5.0, 'k1_all_rwt': 0.09798994974874371, 'k9_all_rmt': 0.06555090655509066, 'k9_all_nmt': 47.0, 'k1_hum_nwt_w': 3.978290366350068, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 77.0, 'k9_hum_rwt_w': 0.08050655811849843, 'k9_all_nwt_w': 14.470827679782904, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.07946278202409605},
        #     {'k9_all_nwt': 343.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.0, 'k1_hum_pwm': -708.8063937958501, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0013202034596319452, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.75, 'k9_hum_pwm': 0.0, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -4.678503885506515, 'k1_all_naa': 1462.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.00314070351758794, 'k1_hum_rwt_w': 0.6043645409317051, 'k9_all_rwt': 0.4783821478382148, 'k1_all_pwm': -5.225708868469609, 'k1_all_nwt': 984.0, 'k1_all_nmt_w': 2.101763907734057, 'k9_all_pwm': -4.682639042349287, 'k9_all_rmt_w': 0.0019132182374547951, 'k9_hum_nwt_w': 0.0, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 0.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.5904981497837, 'k9_hum_pwm_w': 0.0, 'k9_all_naa': 594.0, 'k1_all_nwt_w': 531.2998643147897, 'k9_hum_nal': 3, 'k9_all_nmt_w': 1.371777476255088, 'k9_all_rwt_w': 0.2178419428154019, 'k1_all_pwm_w': -5.476070513697371, 'k1_hum_nwt': 9.0, 'k1_all_rwt': 0.6180904522613065, 'k9_all_rmt': 0.0041841004184100415, 'k9_all_nmt': 3.0, 'k1_hum_nwt_w': 7.252374491180461, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 5.0, 'k9_hum_rwt_w': 0.0, 'k9_all_nwt_w': 156.19267299864316, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.333731070549491},
        #     {'k9_all_nwt': 115.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -708.2289286451407, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.5, 'k9_hum_pwm': -706.4371691759127, 'k9_hum_naa': 2.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -5.055817755870965, 'k1_all_naa': 896.0, 'k1_hum_naa': 10.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.0, 'k1_hum_rwt_w': 0.4271822704658526, 'k9_all_rwt': 0.1603905160390516, 'k1_all_pwm': -712.3092869653881, 'k1_all_nwt': 355.0, 'k1_all_nmt_w': 0.0, 'k9_all_pwm': -3.7662619937683366, 'k9_all_rmt_w': 0.0002005945926510468, 'k9_hum_nwt_w': 0.4735413839891452, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.0715313317847, 'k9_hum_pwm_w': -705.6896532059258, 'k9_all_naa': 415.0, 'k1_all_nwt_w': 247.87924016282224, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.14382632293080055, 'k9_all_rwt_w': 0.08376527404816919, 'k1_all_pwm_w': -711.9501108686591, 'k1_hum_nwt': 6.0, 'k1_all_rwt': 0.22298994974874373, 'k9_all_rmt': 0.001394700139470014, 'k9_all_nmt': 1.0, 'k1_hum_nwt_w': 5.1261872455902315, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 0.0, 'k9_hum_rwt_w': 0.15784712799638173, 'k9_all_nwt_w': 60.059701492537314, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.1557030403032803},
        #     {'k9_all_nwt': 488.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -708.7521415484774, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0007397912220532786, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.75, 'k9_hum_pwm': -706.5549169711412, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -5.9165200352421135, 'k1_all_naa': 1181.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.0037688442211055275, 'k1_hum_rwt_w': 0.5486205336951605, 'k9_all_rwt': 0.6806136680613668, 'k1_all_pwm': -5.363176091925383, 'k1_all_nwt': 1151.0, 'k1_all_nmt_w': 1.1777476255088195, 'k9_all_pwm': -5.198252269616593, 'k9_all_rmt_w': 0.0008042707724216499, 'k9_hum_nwt_w': 0.24151967435549526, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.4394753490158, 'k9_hum_pwm_w': -705.134112629244, 'k9_all_naa': 511.0, 'k1_all_nwt_w': 567.9850746268656, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.576662143826323, 'k9_all_rwt_w': 0.26831229928713224, 'k1_all_pwm_w': -6.285040471271067, 'k1_hum_nwt': 9.0, 'k1_all_rwt': 0.7229899497487438, 'k9_all_rmt': 0.0041841004184100415, 'k9_all_nmt': 3.0, 'k1_hum_nwt_w': 6.5834464043419265, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 6.0, 'k9_hum_rwt_w': 0.08050655811849843, 'k9_all_nwt_w': 192.3799185888738, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.35677454436360906},
        #     {'k9_all_nwt': 497.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -708.710064395281, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 7.755875715074694e-05, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.75, 'k9_hum_pwm': -706.5128398179448, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -7.758386294837598, 'k1_all_naa': 1184.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.000628140703517588, 'k1_hum_rwt_w': 0.5486205336951605, 'k9_all_rwt': 0.6931659693165969, 'k1_all_pwm': -7.474816768623261, 'k1_all_nwt': 1164.0, 'k1_all_nmt_w': 0.12347354138398914, 'k9_all_pwm': -6.623789166428506, 'k9_all_rmt_w': 0.0001741009672065689, 'k9_hum_nwt_w': 0.24151967435549526, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.3973981958195, 'k9_hum_pwm_w': -705.0920354760476, 'k9_all_naa': 513.0, 'k1_all_nwt_w': 569.7964721845318, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.1248303934871099, 'k9_all_rwt_w': 0.2690957536395618, 'k1_all_pwm_w': -8.852206756726668, 'k1_hum_nwt': 9.0, 'k1_all_rwt': 0.7311557788944724, 'k9_all_rmt': 0.001394700139470014, 'k9_all_nmt': 1.0, 'k1_hum_nwt_w': 6.5834464043419265, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 1.0, 'k9_hum_rwt_w': 0.08050655811849843, 'k9_all_nwt_w': 192.9416553595658, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.35791235689983153},
        #     {'k9_all_nwt': 274.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.3333333333333333, 'k1_hum_pwm': -709.2311292771458, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 4.431928980042683e-05, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.8333333333333334, 'k9_hum_pwm': -706.9285441841517, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -7.277066334052013, 'k1_all_naa': 1377.0, 'k1_hum_naa': 10.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.000628140703517588, 'k1_hum_rwt_w': 0.6442786069651741, 'k9_all_rwt': 0.3821478382147838, 'k1_all_pwm': -6.517140190990621, 'k1_all_nwt': 818.0, 'k1_all_nmt_w': 0.07055630936227951, 'k9_all_pwm': -5.423405960775945, 'k9_all_rmt_w': 9.840489450806068e-05, 'k9_hum_nwt_w': 0.4735413839891452, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 1.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.9738268070203, 'k9_hum_pwm_w': -706.1810282141648, 'k9_all_naa': 600.0, 'k1_all_nwt_w': 477.2713704206241, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.07055630936227951, 'k9_all_rwt_w': 0.17210826809278065, 'k1_all_pwm_w': -8.629707267850826, 'k1_hum_nwt': 10.0, 'k1_all_rwt': 0.5138190954773869, 'k9_all_rmt': 0.001394700139470014, 'k9_all_nmt': 1.0, 'k1_hum_nwt_w': 7.73134328358209, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 1.0, 'k9_hum_rwt_w': 0.15784712799638173, 'k9_all_nwt_w': 123.40162822252373, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.29979357438481413},
        #     {'k9_all_nwt': 381.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.0, 'k1_hum_pwm': -709.3154909071001, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0003971690201345943, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.75, 'k9_hum_pwm': 0.0, 'k9_hum_naa': 1.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -712.2743070615036, 'k1_all_naa': 1463.0, 'k1_hum_naa': 9.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.001256281407035176, 'k1_hum_rwt_w': 0.6043645409317051, 'k9_all_rwt': 0.5313807531380753, 'k1_all_pwm': -6.893285071980811, 'k1_all_nwt': 1076.0, 'k1_all_nmt_w': 0.6322930800542741, 'k9_all_pwm': -713.0610657048906, 'k9_all_rmt_w': 0.0, 'k9_hum_nwt_w': 0.0, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 0.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -709.0995952610338, 'k9_hum_pwm_w': 0.0, 'k9_all_naa': 598.0, 'k1_all_nwt_w': 583.662143826323, 'k9_hum_nal': 3, 'k9_all_nmt_w': 0.0, 'k9_all_rwt_w': 0.24194735716624183, 'k1_all_pwm_w': -7.433151064457855, 'k1_hum_nwt': 9.0, 'k1_all_rwt': 0.6758793969849246, 'k9_all_rmt': 0.0, 'k9_all_nmt': 0.0, 'k1_hum_nwt_w': 7.252374491180461, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 2.0, 'k9_hum_rwt_w': 0.0, 'k9_all_nwt_w': 173.4762550881954, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.3666219496396501},
        #     {'k9_all_nwt': 326.0, 'k9_hum_rmt': 0.0, 'k9_hum_rwt': 0.6666666666666666, 'k1_hum_pwm': -709.1435996013165, 'k1_hum_nmt': 0.0, 'k1_all_rmt_w': 0.0037296386955128426, 'k1_hum_nal': 12, 'k1_hum_rwt': 0.8333333333333334, 'k9_hum_pwm': -707.5341616888824, 'k9_hum_naa': 2.0, 'k1_hum_nmt_w': 0.0, 'k9_all_pwm_w': -4.21117347082725, 'k1_all_naa': 1256.0, 'k1_hum_naa': 10.0, 'k1_all_nal': 1592, 'k9_hum_nmt': 0.0, 'k1_all_rmt': 0.015703517587939697, 'k1_hum_rwt_w': 0.6442786069651741, 'k9_all_rwt': 0.45467224546722457, 'k1_all_pwm': -3.732997341163993, 'k1_all_nwt': 877.0, 'k1_all_nmt_w': 5.9375848032564456, 'k9_all_pwm': -3.477356905238717, 'k9_all_rmt_w': 0.003813189662187351, 'k9_hum_nwt_w': 0.55359565807327, 'k1_hum_rmt': 0.0, 'k9_hum_nwt': 2.0, 'k9_hum_rmt_w': 0.0, 'k9_all_nal': 717, 'k1_hum_pwm_w': -708.886297131191, 'k9_hum_pwm_w': -706.2496937905373, 'k9_all_naa': 538.0, 'k1_all_nwt_w': 536.0827679782903, 'k9_hum_nal': 3, 'k9_all_nmt_w': 2.734056987788331, 'k9_all_rwt_w': 0.21578300963800248, 'k1_all_pwm_w': -4.678352288767064, 'k1_hum_nwt': 10.0, 'k1_all_rwt': 0.5508793969849246, 'k9_all_rmt': 0.016736401673640166, 'k9_all_nmt': 12.0, 'k1_hum_nwt_w': 7.73134328358209, 'k9_hum_nmt_w': 0.0, 'k1_all_nmt': 25.0, 'k9_hum_rwt_w': 0.18453188602442336, 'k9_all_nwt_w': 154.71641791044777, 'k1_hum_rmt_w': 0.0, 'k1_all_rwt_w': 0.33673540702153915},
        # ]
        # self._check_features(variants, expected_features)

        # Short test
        variants = variants_L8EA00.sample(2, random_state=42)
        _kalign_features(variants, _FEATURES['kalign'])
        expected_features = [
            {'k1_all_nmt': 0.000000, 'k9_all_nal': 2.000000, 'k9_all_naa': 2.000000, 'k9_hum_nmt': 0.000000, 'k9_all_rwt': 1.000000, 'k9_all_pwm_w': -10.000000, 'k1_all_naa_w': 1.000000, 'k1_hum_rwt': 1.000000, 'k9_hum_pwm_w': -10.000000, 'k9_all_nmt': 0.000000, 'k1_all_naa': 1.000000, 'k9_hum_pwm': -10.000000, 'k9_all_nal_w': 2.000000, 'k1_hum_rmt': 0.000000, 'k9_hum_rwt_w': 1.000000, 'k1_all_nwt_w': 1.000000, 'k9_hum_nmt_w': 0.000000, 'k1_all_nal': 1.000000, 'k9_all_pwm': -10.000000, 'k1_hum_nal_w': 1.000000, 'k9_all_nwt': 2.000000, 'k1_hum_nal': 1.000000, 'k1_all_rmt': 0.000000, 'k1_all_nwt': 1.000000, 'k1_all_pwm': -10.000000, 'k1_hum_naa_w': 1.000000, 'k1_all_rwt': 1.000000, 'k9_all_rmt_w': 0.000000, 'k9_hum_nwt': 2.000000, 'k9_all_rwt_w': 1.000000, 'k1_hum_rmt_w': 0.000000, 'k9_hum_naa': 2.000000, 'k9_hum_rmt_w': 0.000000, 'k1_all_pwm_w': -10.000000, 'k9_hum_nwt_w': 2.000000, 'k9_all_rmt': 0.000000, 'k9_hum_rmt': 0.000000, 'k9_hum_naa_w': 2.000000, 'k1_all_rmt_w': 0.000000, 'k1_hum_nwt_w': 1.000000, 'k1_all_rwt_w': 1.000000, 'k9_hum_nal': 2.000000, 'k9_all_naa_w': 2.000000, 'k9_hum_rwt': 1.000000, 'k1_hum_nmt_w': 0.000000, 'k1_hum_naa': 1.000000, 'k9_all_nmt_w': 0.000000, 'k9_hum_nal_w': 2.000000, 'k1_hum_pwm_w': -10.000000, 'k1_hum_pwm': -10.000000, 'k1_all_nmt_w': 0.000000, 'k9_all_nwt_w': 2.000000, 'k1_hum_nmt': 0.000000, 'k1_all_nal_w': 1.000000, 'k1_hum_nwt': 1.000000, 'k1_hum_rwt_w': 1.000000},
            {'k1_all_nmt': 0.000000, 'k9_all_nal': 2.000000, 'k9_all_naa': 2.000000, 'k9_hum_nmt': 0.000000, 'k9_all_rwt': 1.000000, 'k9_all_pwm_w': -10.000000, 'k1_all_naa_w': 1.000000, 'k1_hum_rwt': 1.000000, 'k9_hum_pwm_w': -10.000000, 'k9_all_nmt': 0.000000, 'k1_all_naa': 1.000000, 'k9_hum_pwm': -10.000000, 'k9_all_nal_w': 2.000000, 'k1_hum_rmt': 0.000000, 'k9_hum_rwt_w': 1.000000, 'k1_all_nwt_w': 1.000000, 'k9_hum_nmt_w': 0.000000, 'k1_all_nal': 1.000000, 'k9_all_pwm': -10.000000, 'k1_hum_nal_w': 1.000000, 'k9_all_nwt': 2.000000, 'k1_hum_nal': 1.000000, 'k1_all_rmt': 0.000000, 'k1_all_nwt': 1.000000, 'k1_all_pwm': -10.000000, 'k1_hum_naa_w': 1.000000, 'k1_all_rwt': 1.000000, 'k9_all_rmt_w': 0.000000, 'k9_hum_nwt': 2.000000, 'k9_all_rwt_w': 1.000000, 'k1_hum_rmt_w': 0.000000, 'k9_hum_naa': 2.000000, 'k9_hum_rmt_w': 0.000000, 'k1_all_pwm_w': -10.000000, 'k9_hum_nwt_w': 2.000000, 'k9_all_rmt': 0.000000, 'k9_hum_rmt': 0.000000, 'k9_hum_naa_w': 2.000000, 'k1_all_rmt_w': 0.000000, 'k1_hum_nwt_w': 1.000000, 'k1_all_rwt_w': 1.000000, 'k9_hum_nal': 2.000000, 'k9_all_naa_w': 2.000000, 'k9_hum_rwt': 1.000000, 'k1_hum_nmt_w': 0.000000, 'k1_hum_naa': 1.000000, 'k9_all_nmt_w': 0.000000, 'k9_hum_nal_w': 2.000000, 'k1_hum_pwm_w': -10.000000, 'k1_hum_pwm': -10.000000, 'k1_all_nmt_w': 0.000000, 'k9_all_nwt_w': 2.000000, 'k1_hum_nmt': 0.000000, 'k1_all_nal_w': 1.000000, 'k1_hum_nwt': 1.000000, 'k1_hum_rwt_w': 1.000000}]
        self._check_features(variants, expected_features)

    # Get Blast

    def testGetBlast(self):
        # blast = _get_blast('P13807', P13807_SEQUENCE, '90')
        # TODO: add some test
        # blast = _get_blast('P13807', P13807_SEQUENCE, '100')
        # TODO: add some test
        blast = _get_blast('L8EA00', L8EA00_SEQUENCE, '90')
        # TODO: add some test
        blast = _get_blast('L8EA00', L8EA00_SEQUENCE, '100')
        self.assertEqual(blast['aas'][0][0], b'M')
        self.assertEqual(blast['aas'][0][-1], b'P')
        self.assertAlmostEqual(blast['evalue'][0][0], 8.65971000e-24)
        self.assertAlmostEqual(blast['evalue'][0][-1], 8.65971000e-24)
        self.assertAlmostEqual(blast['scores'][0][0], 94.7898)
        self.assertAlmostEqual(blast['scores'][0][-1], 94.7898)
        self.assertEqual(len(blast['human']), 1)
        self.assertTrue(blast['human'][0])

    # Kalign features

    # Get Kalign

    def testGetKalign(self):
        # kalign = _get_kalign('P13807', P13807_SEQUENCE, '90')
        # TODO: add some test
        # kalign = _get_kalign('P13807', P13807_SEQUENCE, '100')
        # TODO: add some test
        kalign = _get_kalign('L8EA00', L8EA00_SEQUENCE, '90')
        # TODO: add some test
        kalign = _get_kalign('L8EA00', L8EA00_SEQUENCE, '100')
        self.assertTrue(kalign['human'][0])
        self.assertEqual(kalign['similarity'][0], 1.0)
        self.assertEqual(kalign['aligned_aas'][0][0], b'M')
        self.assertEqual(kalign['names'][0], 'UniRef100_L8EA00')

    # Features distribution

    def testFeaturesDistribution(self):
        variants = variants_L8EA00.sample(20, random_state=42)
        variants[PMUT_FEATURES[0]] = 1.
        variants[PMUT_FEATURES[1]] = 2.
        variants[PMUT_FEATURES[2]] = 3.
        variants['disease'] = True
        for (fname, figure), pmut_feature in zip(
                features_distribution(variants), PMUT_FEATURES):
            self.assertEqual(fname, feature_name(pmut_feature))
        self.assertEqual(
            len(features_distribution(variants, PMUT_FEATURES[:2])), 2)


################
#   utils.py   #
################

class UtilsTests(unittest.TestCase):
    def test_variant_str(self):
        self.assertEqual(variant_str(variants_P13807.iloc[0]), 'P13807 1.M>A')

    def test_load_fasta(self):
        f = io.BytesIO(b'>bla\nAAA\nAAA\nAAA\n\n>blo\nCCC\n')
        self.assertEqual(load_fasta_str(f), {'bla': 'AAAAAAAAA', 'blo': 'CCC'})

    def test_load_fasta_iter(self):
        f = io.BytesIO(b'>bla\nAAA\nAAA\nAAA\n\n>blo\nCCC\n')
        fasta = [('bla', np.repeat(b'A', 9)), ('blo', np.repeat(b'C', 3))]
        for (title, seq), (title2, seq2) in zip(fasta, load_fasta_iter(f)):
            self.assertEqual(title, title2)
            np.testing.assert_array_equal(seq, seq2)

    def testAllVariants(self):
        sequence = 'MPLNRTLSMSSLPGLEDWEDEFDLEN'
        variants = all_variants('P13807', sequence)
        self.assertEqual(len(variants), 19*len(sequence))
        # TODO: need more tests

    def testParseVariant(self):
        self.assertEqual(parse_variant('p.Gly207Arg'),
                         {'position': 207, 'wt': 'G', 'mt': 'R'})


################
#   data.py   #
################

class dataTests(unittest.TestCase):
    def test_data_exist(self):
        self.assertTrue(os.path.isdir(_PROJECT_DIR))
        self.assertTrue(os.path.isdir(_DATA_DIR))
        self.assertTrue(os.path.isdir(_VENDOR_DIR))
        self.assertTrue(os.path.isfile(_UNIREF100_2_UNIPROT_FILE))
        self.assertTrue(os.path.isfile(_UNIREF50_2_UNIPROT_FILE))
        self.assertTrue(os.path.isfile(_NETWORK_PARAM_FILE))
        self.assertTrue(os.path.isfile(_BLASTDBCMD))
        for ext in ['phr', 'pin', 'pog', 'psd', 'psi', 'psq']:
            self.assertTrue(os.path.isfile(_HUMAN_UNIREF100 + '.' + ext))

    # Fasta, Blast, Fa, Kalign data
    def testPaths(self):
        for protein_id, db in itertools.product(['P13807', 'L8EA00'], _DBS):
            self.assertTrue(path_fasta(protein_id, db))
            self.assertTrue(path_blast(protein_id, db))
            self.assertTrue(path_fa(protein_id, db))
            self.assertTrue(path_kalign(protein_id, db))

        for db in _DBS:
            with self.assertRaises(Exception):
                path_fasta('invented', db)
            with self.assertRaises(Exception):
                path_blast('invented', db)
            with self.assertRaises(Exception):
                path_fa('invented', db)
            with self.assertRaises(Exception):
                path_kalign('invented', db)

    # UniProt -> UniRef100
    def testUniref100(self):
        self.assertEqual(uniref100('A0A023I7H5'), 'UniRef100_A0A023I7H5')
        self.assertEqual(uniref100('Q8NI08-5'), 'UniRef100_A0A023IN41')
        self.assertEqual(uniref100('G3IFX0'), 'UniRef100_A0A024QZ42')
        self.assertEqual(uniref100('invented'), 'invented')

    # UniProt -> UniRef50
    def testUniref50(self):
        self.assertEqual(uniref50('Q86TX2'), 'UniRef50_P49753')
        self.assertEqual(uniref50('P49753'), 'UniRef50_P49753')
        self.assertEqual(uniref50('P48449'), 'UniRef50_P48449')
        self.assertEqual(uniref50('invented'), 'invented')

    # Protein sequences
    def testGetProteinSequences(self):
        sequences = get_protein_sequences(['P13807', 'L8EA00'])
        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences['P13807'], P13807_SEQUENCE)
        self.assertEqual(sequences['L8EA00'], L8EA00_SEQUENCE)


#############
#   io.py   #
#############

class IOTests(unittest.TestCase):
    def test_read_humsavar(self):
        # Test humsavar file is the standard humsavar file
        # with an erroneous line added
        with open(os.path.join(TEST_DIR, 'humsavar_2016_02.txt'), 'rt') as f:
            with self.assertLogs(level='WARNING'):
                variants = read_humsavar_txt(f)
                self.assertEqual(len(variants), 65090)
                self.assertAlmostEqual(
                    variants.position.mean(), 553.9992625595330)
                self.assertEqual(set(variants.disease), set([True, False]))


######################
#   amino_acids.py   #
######################

class AminoAcidTests(unittest.TestCase):
    def test_amino_acids(self):
        self.assertEqual(set(AMINO_ACIDS), set('ACDEFGHIKLMNPQRSTVWY'))

    def test_amino_acid_names(self):
        self.assertEqual(aa3('A'), 'Ala')
        self.assertEqual(aa3('U'), 'Sec')
        self.assertEqual(aa1('Trp'), 'W')
        self.assertEqual(aa1('Ter'), 'X')
        self.assertEqual(aa_name('R'), 'Arginine')
        self.assertEqual(aa_name('M'), 'Methionine')
        self.assertLessEqual(set(AMINO_ACIDS), set(AA3.keys()))
        self.assertLessEqual(set(AMINO_ACIDS), set(AA_NAME.keys()))
        self.assertSetEqual(set(AA3.values()), set(AA1.keys()))
        self.assertEqual(set(AA1.values()), set(AA3.keys()))
        self.assertEqual(set(AA_NAME.keys()), set(AA3.keys()))

    def test_aa_distance(self):
        self.assertEqual(aa_distance('A', 'A'), 0)
        self.assertEqual(aa_distance('T', 'P'), 1)
        self.assertEqual(aa_distance('V', 'R'), 2)
        self.assertEqual(aa_distance('E', 'C'), 3)
        for aa, aa2 in itertools.product(AA_DISTANCE, repeat=2):
            self.assertEqual(aa_distance(aa, aa2), aa_distance(aa2, aa))


############
#   main   #
############

if __name__ == '__main__':
    unittest.main()


#     def test_sanitize_sequence(self):
#         assert sanitize_sequence('ACDEFGHI\rKL\nMNPQRSTVWY') == 'ACDEFGHIKLMNPQRSTVWY'
#         assert sanitize_sequence('ACDEFG  HIKLMNPQRS  TVWY') == 'ACDEFGHIKLMNPQRSTVWY'
#         assert sanitize_sequence('\r\rACDEFGHIKLMNPQRSTVWY') == 'ACDEFGHIKLMNPQRSTVWY'
#         assert sanitize_sequence('ACDEFGHIKLMNP\tQRSTVWY') == 'ACDEFGHIKLMNPQRSTVWY'
#
