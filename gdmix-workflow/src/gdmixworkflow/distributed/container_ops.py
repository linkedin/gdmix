import fileinput
from gdmixworkflow.common.utils import gen_random_string
from gdmixworkflow.distributed import resource
import os
from os.path import join as path_join
import shutil


def load_launcher_from_file(src_file, name_placeholder, name):
    """ Load launcher's commponent yaml file and update launcher name """
    dest_file = shutil.copyfile(src_file, gen_random_string(20) + ".tmp.yaml")
    with fileinput.FileInput(dest_file, inplace=True) as f:
        for line in f:
            line = line.replace(name_placeholder, name)
            print(line, end='')
    from kfp import components
    component = components.load_component_from_file(dest_file)
    os.remove(dest_file)
    return component


def gdmix_tfjob_op(
        name,
        cmd,
        namespace,
        secretName,
        image="linkedin/gdmix",
        workerType="cpu",
        needChief=False,
        psNum=0,
        evaluatorNum=0,
        workerNum=2,
        memorySize='1G',
        ttlSecondsAfterFinished=-1,
        tfjobTimeoutMinutes=1440,
        deleteAfterDone=False):
    """
    This function prepares params for launch_tfjob.py, as defined in the
    templates of ../launcher/tfjob/tfjob_component.yaml, which specifies to
    execute the launch_tfjob.py once the container is ready. In the container,
    The launch_tfjob.py assemble all params to form a deployable YAML file to
    launch the actual TFJob.
    """
    componentPath = path_join(resource.__path__[0],
                             "tfjob_component.yaml")
    name_placeholder = "TFJob-launcher-name"
    tfjob_launcher_op = load_launcher_from_file(
        componentPath, name_placeholder, name)

    # Container spec of a TFJob worker
    containerSpec = {
        "command": ["sh", "-c", "{}".format(cmd)],
        "image": "{}".format(image),
        "resources": {"limits": {"memory": "{}i".format(memorySize.upper())},
                      "requests": {"memory": "{}i".format(memorySize.upper())}},
        "name": "tensorflow",
        "volumeMounts": [{"name": "shared-data",
                          "mountPath": "/var/tmp"},
                         {"name": "dt",
                          "mountPath": "/var/dt"}]
    }

    # Volume spec for model code and secrets
    volumeSpec = [{"name": "shared-data",
                   "emptyDir": {"sizeLimit": "10Mi"}
                  },
                  {"name": "dt",
                  "secret": {"secretName": secretName,
                             "defaultMode": 256}
                  }]

    if (workerType == "gpu"):
        containerSpec["resources"]["limits"]["nvidia.com/gpu"] = 1

    workerSpecTemplate = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "tfReplicaType": "WORKER",
        "template": {
            "spec": {
                "containers": [containerSpec],
                "volumes": volumeSpec
            }
        }
    }

    chief = ps = evaluator = worker = {}

    if needChief:
        chief = workerSpecTemplate.copy()
        chief["tfReplicaType"] = "MASTER"
    if psNum > 0:
        ps = workerSpecTemplate.copy()
        ps["tfReplicaType"] = "PS"
        ps["replicas"] = psNum
    if evaluatorNum > 0:
        evaluator = workerSpecTemplate.copy()
        evaluator["tfReplicaType"] = "EVALUATOR"
        evaluator["replicas"] = evaluatorNum
    if workerNum > 0:
        worker = workerSpecTemplate.copy()
        worker["replicas"] = workerNum

    op = tfjob_launcher_op(
        name=name,
        namespace=namespace,
        ttl_seconds_after_finished=ttlSecondsAfterFinished,
        ps_spec=ps,
        worker_spec=worker,
        chief_spec=chief,
        evaluator_spec=evaluator,
        tfjob_timeout_minutes=tfjobTimeoutMinutes,
        delete_finished_tfjob=deleteAfterDone
    )
    return op


def gdmix_sparkjob_op(
        name,
        mainClass,
        arguments,
        secretName,
        mainApplicationFile="local:///opt/spark/jars/gdmix-data-all_2.11.jar",
        image="linkedin/gdmix",
        namespace="default",
        serviceAccount="default",
        driverCores=1,
        driverMemory='2g',
        executorCores=2,
        executorInstances=2,
        executorMemory='1g',
        sparkApplicationTimeoutMinutes=1440,
        deleteAfterDone=False):
    """
    This function prepares params for launch_sparkapplication.py as defined in
    the templates of ../launcher/sparkapplication/sparkapplication_component.yaml,
    which specifies to execute the launch_tfjob.py once the container is ready.
    In the container, the launch_tfjob.py assemble all params to form a deployable
    YAML file to launch the actual spark application.
    """
    componentPath = path_join(resource.__path__[0],
                             "sparkapplication_component.yaml")
    name_placeholder = "SparkApplication-launcher-name"
    spark_application_launcher_op = load_launcher_from_file(
        componentPath, name_placeholder, name)

    # Driver spec for the spark application
    # Note: secret for now is pre-craeted and will be valid for 7 days
    driverSpec = {
        "cores": driverCores,
        "memory": driverMemory,
        "secrets": [
            {
                "name": secretName,
                "path": "/var/tmp/{}".format(secretName),
                "secretType": "HadoopDelegationToken"
            }
        ],
        "serviceAccount": serviceAccount
    }

    # Executor spec for the spark application
    executorSpec = {
        "cores": executorCores,
        "instances": executorInstances,
        "memory": executorMemory
    }

    return spark_application_launcher_op(
        name=name,
        namespace=namespace,
        image=image,
        main_class=mainClass,
        arguments=arguments,
        main_application_file=mainApplicationFile,
        driver_spec=driverSpec,
        executor_spec=executorSpec,
        sparkapplication_timeout_minutes=sparkApplicationTimeoutMinutes,
        delete_finished_sparkapplication=deleteAfterDone
    )


def no_op(msg):
    """ Empty op as a placeholder on workflow to handle dependency.
    """
    from kfp import dsl
    return dsl.ContainerOp(
        name="{}-{}".format(msg, gen_random_string()),
        image="alpine",
        command=['sh', '-c'],
        arguments=["echo {}".format(msg)]
    )
