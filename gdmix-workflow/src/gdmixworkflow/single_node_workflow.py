from gdmixworkflow.common.utils import *
from gdmixworkflow.common.constants import *
from gdmixworkflow.fixed_effect_workflow_generator \
    import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator \
    import RandomEffectWorkflowGenerator
import json
import os
from os.path import join as path_join
import shutil


def create_output_dirs(gdmix_config_obj):
    """ Create output directories """

    def create_subdirs(root_dir):
        if os.path.isdir(root_dir):
            shutil.rmtree(root_dir)
        os.makedirs(root_dir)
        for sub_dir_name in [MODELS, METRIC, TRAINING_SCORES, VALIDATION_SCORES]:
            sub_dir = path_join(root_dir, sub_dir_name)
            os.makedirs(sub_dir)

    output_dir = gdmix_config_obj.output_dir
    if hasattr(gdmix_config_obj, FIXED_EFFECT_CONFIG):
        root_dir = path_join(output_dir, gdmix_config_obj.fixed_effect_config.name)
        create_subdirs(root_dir)

    if hasattr(gdmix_config_obj, RANDOM_EFFECT_CONFIG):
        for re_config in gdmix_config_obj.random_effect_config:
            root_dir = path_join(output_dir, re_config.name)
            create_subdirs(root_dir)

            num_partitions = re_config.num_partitions
            for score_output_name in [TRAINING_SCORES, VALIDATION_SCORES]:
                sub_dir = path_join(root_dir, score_output_name)
                for idx in range(num_partitions):
                    os.makedirs(path_join(sub_dir, "partitionId={}".format(idx)))


def run_gdmix_single_node(gdmix_config_file, jar_path):
    """ Run gdmix jobs locally including:
        - fixed-effect jobs
        - random-effect jobs
    """
    gdmix_config_obj = json_config_file_to_obj(gdmix_config_file)
    output_dir = gdmix_config_obj.output_dir

    create_output_dirs(gdmix_config_obj)

    if hasattr(gdmix_config_obj, FIXED_EFFECT_CONFIG):
        fe_workflow = FixedEffectWorkflowGenerator(gdmix_config_obj, jar_path=jar_path)
        fe_workflow.run()

    if hasattr(gdmix_config_obj, RANDOM_EFFECT_CONFIG):
        re_workflow = RandomEffectWorkflowGenerator(gdmix_config_obj, jar_path=jar_path)
        re_workflow.run()

    return output_dir