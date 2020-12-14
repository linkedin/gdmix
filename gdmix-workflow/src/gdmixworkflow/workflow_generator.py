from abc import abstractmethod

from gdmixworkflow.common.constants import *
from gdmixworkflow.common.utils import rm_backslash
from gdmixworkflow.distributed.container_ops import gdmix_tfjob_op, gdmix_sparkjob_op
from gdmixworkflow.single_node.local_ops import get_tfjob_cmd, get_sparkjob_cmd, run_cmd, get_param_list


class WorkflowGenerator(object):
    def __init__(self, gdmix_config_obj, jar_path="", namespace="",
                 secret_name="", image="", service_account="", job_suffix=""):
        """ Init to generate full gdmix fix-effect or random-effect workflow.
            the config obj is a named tuple and its name/value pair can be passed
            as parameters to tensorflow jobs, need to handle itermediate and output paths
        """
        self.gdmix_config_obj = gdmix_config_obj
        self.namespace = namespace
        self.secret_name = secret_name
        self.image = image
        self.service_account = service_account
        self.suffix = job_suffix
        self.jar_path = jar_path

    def tip(self, job_name, cmd):
        """ single-node: show message for current job."""
        print(f"""
------------------------
    {self.get_name(job_name)}
------------------------
Executing cmd:\n  {' '.join(cmd)}\n
              """)

    def get_name(self, name):
        """ Append suffix to name and transform to lowercase. k8s requires
        job name must be lower case alphanumeric characters, '-' or '.', and
        full job name length should < 64. Here limit to 40 for potential
        suffix for subsequent jobs.
        """
        LENGTH = 40
        full_name = f"{name}-{self.suffix}" if self.suffix else name
        return full_name.lower()[:LENGTH]

    def get_tfjob_config(self, extra_config):
        """ get tf job config for tf job container. """
        return {
            "namespace": self.namespace,
            "secretName": self.secret_name,
            "image": self.image,
            **self.gdmix_config_obj.tfjob_config,
            **extra_config}

    def get_sparkjob_config(self, extra_config):
        """ get spark job config for spark job container. """
        return {
            "namespace": self.namespace,
            "secretName": self.secret_name,
            "image": self.image,
            "serviceAccount": self.service_account,
            **self.gdmix_config_obj.spark_config,
            **extra_config}

    @abstractmethod
    def get_job_sequence(self):
        pass

    def gen_workflow(self):
        """ Generate gdmix fixed/random effect workflow from given job sequence.
            Returns:
                start_op: first job of the workflow
                end_op: last job of the workflow
                trainScorePath: training data score result path
                validationScorePath: validation data score result path
        """
        jobs = self.get_job_sequence()

        start_op = prev_op = current_op = None
        for (job_type, job_name, class_name, params) in jobs:
            if job_type == GDMIX_SPARKJOB:
                extra_config = {
                    "name": self.get_name(job_name),
                    "mainClass": class_name,
                    "arguments": ' '.join(get_param_list(params))
                }
                sparkjob_config = self.get_sparkjob_config(extra_config)
                current_op = gdmix_sparkjob_op(**sparkjob_config)
            elif job_type == GDMIX_TFJOB:
                extra_config = {
                    "name": self.get_name(job_name),
                    "cmd": ' '.join(get_tfjob_cmd(params)),
                }
                tfjob_config = self.get_tfjob_config(extra_config)
                current_op = gdmix_tfjob_op(**tfjob_config)

            if not prev_op:
                start_op = prev_op = current_op
            else:
                current_op.after(prev_op)
                prev_op = current_op

        return start_op, current_op

    def run(self):
        """ Run all gdmix fixed/random effect jobs at local from given job sequence.
        """
        jobs = self.get_job_sequence()

        for (job_type, job_name, class_name, params) in jobs:
            if job_type == GDMIX_SPARKJOB:
                cmd = get_sparkjob_cmd(class_name, rm_backslash(params), self.jar_path)
            elif job_type == GDMIX_TFJOB:
                cmd = get_tfjob_cmd(params)
            self.tip(job_name, cmd)
            run_cmd(cmd)
