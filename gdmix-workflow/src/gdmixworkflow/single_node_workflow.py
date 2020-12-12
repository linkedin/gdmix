import os
import shutil
from os.path import join as path_join

from gdmixworkflow.common.utils import *
from gdmixworkflow.common.constants import *
from gdmixworkflow.fixed_effect_workflow_generator \
    import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator \
    import RandomEffectWorkflowGenerator


def create_subdirs(parent_dir):
    if os.path.isdir(parent_dir):
        shutil.rmtree(parent_dir)
    os.makedirs(parent_dir)
    for sub_dir_name in (MODELS, METRIC, TRAINING_SCORES, VALIDATION_SCORES):
        os.makedirs(path_join(parent_dir, sub_dir_name))


def run_gdmix_single_node(gdmix_config_file, jar_path):
    """ Run gdmix jobs locally including:
        - fixed-effect jobs
        - random-effect jobs
    """
    gdmix_config_obj = json_config_file_to_obj(gdmix_config_file)
    output_dir = gdmix_config_obj.output_dir

    if not hasattr(gdmix_config_obj, FIXED_EFFECT_CONFIG):
        raise ValueError(f"Need to define {FIXED_EFFECT_CONFIG}")
    fe_workflow = FixedEffectWorkflowGenerator(gdmix_config_obj, jar_path=jar_path)
    root_dir = path_join(output_dir, fe_workflow.fixed_effect_name)
    create_subdirs(root_dir)
    fe_workflow.run()

    if hasattr(gdmix_config_obj, RANDOM_EFFECT_CONFIG):
        for name, re_config in gdmix_config_obj.random_effect_config.items():
            root_dir = path_join(output_dir, name)
            create_subdirs(root_dir)
            num_partitions = re_config['num_partitions']
            for score_output_name in (TRAINING_SCORES, VALIDATION_SCORES):
                sub_dir = path_join(root_dir, score_output_name)
                for idx in range(num_partitions):
                    os.makedirs(path_join(sub_dir, f"partitionId={idx}"))
        re_workflow = RandomEffectWorkflowGenerator(gdmix_config_obj, jar_path=jar_path, prev_model_name=fe_workflow.fixed_effect_name)
        re_workflow.run()

    return output_dir
