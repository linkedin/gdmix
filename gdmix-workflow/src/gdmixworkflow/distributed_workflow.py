from gdmixworkflow.common.utils import json_config_file_to_obj, gen_random_string
from gdmixworkflow.common.constants import *
from gdmixworkflow.distributed.container_ops import no_op
from gdmixworkflow.fixed_effect_workflow_generator \
    import FixedEffectWorkflowGenerator
from gdmixworkflow.random_effect_workflow_generator \
    import RandomEffectWorkflowGenerator
import kfp.dsl as dsl


@dsl.pipeline()
def gdmix_distributed_workflow(gdmix_config_file, namespace, secret_name, image, service_account):
    """ Generate gdmix kubeflow pipeline using Kubeflow pipeline python DSL( kfp.dsl).
    """

    gdmix_config_obj = json_config_file_to_obj(gdmix_config_file)

    current_op = no_op("GDMix-training-start")
    suffix = gen_random_string()

    if hasattr(gdmix_config_obj, FIXED_EFFECT_CONFIG):
        fe_tip_op = no_op("fixed-effect-training-start")
        fe_tip_op.after(current_op)
        fe_workflow = FixedEffectWorkflowGenerator(gdmix_config_obj,
                                                   namespace=namespace,
                                                   secret_name=secret_name,
                                                   image=image,
                                                   service_account=service_account,
                                                   job_suffix=suffix)
        fe_start_op, current_op = fe_workflow.gen_workflow()
        fe_start_op.after(fe_tip_op)

    if hasattr(gdmix_config_obj, RANDOM_EFFECT_CONFIG):
        re_tip_op = no_op("random-effect-training-start")
        re_tip_op.after(current_op)
        re_workflow = RandomEffectWorkflowGenerator(gdmix_config_obj,
                                                   namespace=namespace,
                                                   secret_name=secret_name,
                                                   image=image,
                                                   service_account=service_account,
                                                   job_suffix=suffix)
        re_start_op, _ = re_workflow.gen_workflow()
        re_start_op.after(re_tip_op)