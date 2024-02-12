from .semisup import ImpreciseSemiSupervisedLearning
from .noisy_label import ImpreciseNoisyLabelLearning
from .partial_label import ImprecisePartialLabelLearning, RCRPartialLabelLearning, LWSPartialLabelLearning, CCPartialLabelLearning, ProdenPartialLabelLearning
from .partial_noisy_ulb import ImpPartialNoisyUnlabeledLearning
from .multi_ins_label import ImpreciseMultipleInstanceLearning, CountLossMultipleInstanceLearning, UUMMultipleInstanceLearning, GatedAttnMultipleInstanceLearning, AttnMultipleInstanceLearning
from .proportion_label import ImpreciseProportionLabelLearning, CountLossProportionLabelLearning, UUMProportionLabelLearning, LLPVATProportionLabelLearning
from .pos_ulb import UnbiasedPositiveUnLabeledLearning, NonNegativePositiveUnlabeledLearning, CVIRPositiveUnlabeled, LabelDistPositiveUnlabeledLearning, CountLossPositiveUnlabeledLearning, VariationalPositiveUnlabeledLearning, ImprecisePositiveUnlabeledLearning
from .ulb_ulb import ImpreciseUnlabeledUnlabeledLearning, UULearnUnlabeledUnlabeledLearning
from .sim_dsim_ulb import ImpreciseSimilarDisimilarUnlabeledLearning, RsikSDPairSimilarDisimilarUnlabeledLearning
from .pair_comp import ImprecisePairComparisonLearning, PCompUnbiasedPairComparisonLearning, PCompCorrectedPairComparisonLearning, PCompTeacherPairComparisonLearning, RankPruningPairComparisonLearning
from .pair_sim import ImprecisePairSimilarityLearning, UUMPairSimilarityLearning, RsikSDPairSimilarityLearning
from .pos_conf import ImprecisePositiveConfidenceLearning, PconfPositiveConfidenceLearning
from .sim_conf import ImpreciseSimilarConfidenceLearning, SconfUnbiasedPairComparisonLearning, SconfCorrectedPairComparisonLearning
from .conf_diff import ImpreciseConfidenceDifferenceLearning, ConfDiffUnbiasedConfidenceDifferenceLearning, ConfDiffCorrectedConfidenceDifferenceLearning

# from .mixture import SemiSupervisedNoisyPartialLearning
# from .noisy_label_v2 import NoisyLabelLearning

name2alg = {
    'imp_semisup': ImpreciseSemiSupervisedLearning,
    'imp_noisy_label': ImpreciseNoisyLabelLearning,
    'imp_partial_noisy_ulb': ImpPartialNoisyUnlabeledLearning,

    'imp_partial_label': ImprecisePartialLabelLearning,
    'rcr_partial_label': RCRPartialLabelLearning,
    # 'pico_partial_label': PiCOPartialLabelLearning,
    'lws_partial_label': LWSPartialLabelLearning,
    'cc_partial_label': CCPartialLabelLearning,
    'proden_partial_label': ProdenPartialLabelLearning,
    
    'imp_multi_ins': ImpreciseMultipleInstanceLearning,
    'count_loss_multi_ins': CountLossMultipleInstanceLearning,
    'uum_multi_ins': UUMMultipleInstanceLearning,
    'attn_multi_ins': AttnMultipleInstanceLearning,
    'gated_attn_multi_ins': GatedAttnMultipleInstanceLearning,
    
    'imp_proportion': ImpreciseProportionLabelLearning,
    'count_loss_proportion': CountLossProportionLabelLearning,
    'uum_proportion': UUMProportionLabelLearning,
    'llp_vat_proportion': LLPVATProportionLabelLearning,
    
    'upu_pos_ulb': UnbiasedPositiveUnLabeledLearning,
    'nnpu_pos_ulb': NonNegativePositiveUnlabeledLearning,
    'cvir_pos_ulb': CVIRPositiveUnlabeled,
    'dist_pu_pos_ulb': LabelDistPositiveUnlabeledLearning,
    'count_loss_pos_ulb': CountLossPositiveUnlabeledLearning,
    'var_pu_pos_ulb': VariationalPositiveUnlabeledLearning,
    'imp_pos_ulb': ImprecisePositiveUnlabeledLearning,
    
    'uulearn_ulb_ulb': UULearnUnlabeledUnlabeledLearning,
    'imp_ulb_ulb': ImpreciseUnlabeledUnlabeledLearning,
    
    'imp_sim_dsim_ulb': ImpreciseSimilarDisimilarUnlabeledLearning,
    'risksd_sim_dsim_ulb': RsikSDPairSimilarDisimilarUnlabeledLearning,
    
    'imp_pair_comp': ImprecisePairComparisonLearning,
    'pcomp_unbiased_pair_comp': PCompUnbiasedPairComparisonLearning,
    'pcomp_relu_pair_comp': PCompCorrectedPairComparisonLearning,
    'pcomp_abs_pair_comp': PCompCorrectedPairComparisonLearning,
    'pcomp_teacher_pair_comp': PCompTeacherPairComparisonLearning,
    'rank_pruning_pair_comp': RankPruningPairComparisonLearning,
    
    'imp_pair_sim': ImprecisePairSimilarityLearning,
    'uum_pair_sim': UUMPairSimilarityLearning,
    'risksd_pair_sim': RsikSDPairSimilarityLearning,
    
    'imp_pos_conf': ImprecisePositiveConfidenceLearning,
    'pconf_pos_conf': PconfPositiveConfidenceLearning,
    
    'imp_sim_conf': ImpreciseSimilarConfidenceLearning,
    'sconf_unbiased_sim_conf': SconfUnbiasedPairComparisonLearning,
    'sconf_relu_sim_conf': SconfCorrectedPairComparisonLearning,
    'sconf_abs_sim_conf': SconfCorrectedPairComparisonLearning,
    'sconf_nn_abs_sim_conf': SconfCorrectedPairComparisonLearning,
    'sconf_nn_relu_sim_conf': SconfCorrectedPairComparisonLearning,
    
    'imp_conf_diff': ImpreciseConfidenceDifferenceLearning,
    'confdiff_unbiased_conf_diff': ConfDiffUnbiasedConfidenceDifferenceLearning,
    'confdiff_relu_conf_diff': ConfDiffCorrectedConfidenceDifferenceLearning,
    'confdiff_abs_conf_diff': ConfDiffCorrectedConfidenceDifferenceLearning,
    
    # 'mixture': SemiSupervisedNoisyPartialLearning
}