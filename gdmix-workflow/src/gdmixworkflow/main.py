import argparse
from functools import partial, update_wrapper
from gdmixworkflow.common.constants import SINGLE_NODE, DISTRIBUTED
from gdmixworkflow.distributed_workflow import gdmix_distributed_workflow
from gdmixworkflow.single_node_workflow import run_gdmix_single_node
import os
import sys


def str2bool(v):
    """
    handle argparse can't parse boolean well.
    ref: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse/36031646
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.lower() == 'true'
    else:
        raise argparse.ArgumentTypeError('Boolean or string value expected.')


def get_parser():
    """ Creates an argument parser.  """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='path to gdmix config')
    parser.add_argument(
        '--mode',
        type=str,
        default=SINGLE_NODE,
        help='distributed or single_node')
    parser.add_argument(
        '--jar_path',
        type=str,
        default="gdmix-data-all_2.11.jar",
        help='local path to the gdmix-data jar for GDMix processing'
        'intermediate data, single_node only')
    parser.add_argument(
        '--workflow_name',
        type=str,
        default="gdmix-workflow",
        help='name for the generated zip file to upload to'
        'Kubeflow Pipeline, distributed mode only')
    parser.add_argument(
        '--namespace',
        type=str,
        default="default",
        help='Kubernetes namespace, distributed mode only')
    parser.add_argument(
        '--secret_name',
        type=str,
        default="default",
        help='secret name to access storage, distributed mode only')
    parser.add_argument(
        '--image',
        type=str,
        default="linkedin/gdmix",
        help='image used to launch gdmix jobs on Kubernetes, '
        'distributed mode only')
    parser.add_argument(
        '--service_account',
        type=str,
        default="default",
        help='service account to launch spark job, distributed mode only')
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    if args.mode == SINGLE_NODE:
        try:
            outputDir = run_gdmix_single_node(args.config_path, args.jar_path)
        except RuntimeError as err:
            print(str(err))
            sys.exit(1)

        print("""
------------------------
GDMix training is finished, results are saved to {}.
            """.format(outputDir))

    elif args.mode == DISTRIBUTED:
        if not args.namespace:
            print("ERROR: --namespace is required for distributed mode")
            sys.exit(1)

        def wrapped_partial(func, *args, **kwargs):
            partial_func = partial(func, *args, **kwargs)
            update_wrapper(partial_func, func)
            return partial_func

        func = wrapped_partial(
            gdmix_distributed_workflow,
            args.config_path,
            args.namespace,
            args.secret_name,
            args.image,
            args.service_account)

        outputFileName = args.workflow_name + ".zip"

        import kfp.compiler as compiler
        compiler.Compiler().compile(func, outputFileName)
        print("Workflow file is saved to {}".format(outputFileName))

    else:
        print("ERROR: --mode={} isn't supported.".format(args.mode))
        sys.exit(1)


if __name__ == "__main__":
    main()
