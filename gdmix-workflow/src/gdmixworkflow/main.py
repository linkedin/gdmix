import sys
from functools import partial, update_wrapper
from typing import NamedTuple

from smart_arg import arg_suite

from gdmixworkflow.common.constants import SINGLE_NODE, DISTRIBUTED
from gdmixworkflow.distributed_workflow import gdmix_distributed_workflow
from gdmixworkflow.single_node_workflow import run_gdmix_single_node


@arg_suite
class FlowArgs(NamedTuple):
    """ Creates gdmix workflow.  """
    config_path: str  # path to gdmix config
    mode: str = SINGLE_NODE  # distributed or single_node
    jar_path: str = "gdmix-data-all_2.11.jar"  # local path to the gdmix-data jar for GDMix processing intermediate data, single_node only
    workflow_name: str = "gdmix-workflow"  # name for the generated zip file to upload to Kubeflow Pipeline, distributed mode only
    namespace: str = "default"  # Kubernetes namespace, distributed mode only
    secret_name: str = "default"  # secret name to access storage, distributed mode only
    image: str = "linkedin/gdmix"  # image used to launch gdmix jobs on Kubernetes, distributed mode only
    service_account: str = "default"  # service account to launch spark job, distributed mode only


def main():
    args: FlowArgs = FlowArgs.__from_argv__()

    if args.mode == SINGLE_NODE:
        try:
            output_dir = run_gdmix_single_node(args.config_path, args.jar_path)
        except RuntimeError as err:
            print(str(err))
            sys.exit(1)

        print(f"""
------------------------
GDMix training is finished, results are saved to {output_dir}.
            """)

    elif args.mode == DISTRIBUTED:
        if not args.namespace:
            print("ERROR: --namespace is required for distributed mode")
            sys.exit(1)

        wrapper = partial(
            gdmix_distributed_workflow,
            args.config_path,
            args.namespace,
            args.secret_name,
            args.image,
            args.service_account)
        update_wrapper(wrapper, gdmix_distributed_workflow)

        output_file_name = args.workflow_name + ".zip"

        import kfp.compiler as compiler
        compiler.Compiler().compile(wrapper, output_file_name)
        print(f"Workflow file is saved to {output_file_name}")

    else:
        print(f"ERROR: --mode={args.mode} isn't supported.")
        sys.exit(1)


if __name__ == "__main__":
    main()
