import argparse
import datetime
from distutils.util import strtobool
import json
import os
import logging
import yaml
import launch_crd

from kubernetes import client as k8s_client
from kubernetes import config

def yamlOrJsonStr(str):
    if str == "" or str == None:
        return None
    return yaml.safe_load(str)

SparkJobGroup = "sparkoperator.k8s.io"
SparkJobPlural = "sparkapplications"

class SparkApplication(launch_crd.K8sCR):
    def __init__(self, version="v1beta2", client=None):
        super(SparkApplication, self).__init__(SparkJobGroup, SparkJobPlural, version, client)

    def is_expected_conditions(self, inst, expected_conditions):
        conditions = inst.get('status', {}).get("applicationState")
        if not conditions:
            return False, ""
        return conditions["state"] == "COMPLETED", conditions["state"]

def main(argv=None):
    parser = argparse.ArgumentParser(description='SparkApplication launcher')
    parser.add_argument('--name', type=str,
                        help='SparkApplication name.')
    parser.add_argument('--namespace', type=str,
                        default='k8s-spark',
                        help='SparkApplication namespace.')
    parser.add_argument('--version', type=str,
                        default='v1beta2',
                        help='SparkApplication version.')
    parser.add_argument('--restartPolicy', type=str,
                        default="Never",
                        help='Defines the policy when the spark application fails.')
    parser.add_argument('--image', type=str,
                        default="",
                        help='spark image.')
    parser.add_argument('--mainClass', type=str,
                        default="",
                        help='spark job main class.')
    parser.add_argument('--arguments', type=str,
                        default="",
                        help='spark job auguments, separate by , .')
    parser.add_argument('--mainApplicationFile', type=str,
                        default="",
                        help='spark job main file.')
    parser.add_argument('--sparkVersion', type=str,
                        default="3.0.0-SNAPSHOT",
                        help='spark version.')
    parser.add_argument('--driverSpec', type=yamlOrJsonStr,
                        default={},
                        help='SparkApplication driver spec.')
    parser.add_argument('--executorSpec', type=yamlOrJsonStr,
                        default={},
                        help='SparkApplication executor spec.')
    parser.add_argument('--deleteAfterDone', type=strtobool,
                        default=True,
                        help='When sparkApplicaiton done, delete the SparkApplication automatically if it is True.')
    parser.add_argument('--sparkApplicationTimeoutMinutes', type=int,
                        default=60*24,
                        help='Time in minutes to wait for the SparkApplication to reach end')

    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    logging.info('Generating sparkApplication template.')

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    sparkapp = SparkApplication(version=args.version, client=api_client)
    inst = {
        "apiVersion": "%s/%s" % (SparkJobGroup, args.version),
        "kind": "SparkApplication",
        "metadata": {
            "name": args.name,
            "namespace": args.namespace,
        },
        "spec": {
            "type": "Scala",
            "mode": "cluster",
            "image": args.image,
            "imagePullPolicy": "IfNotPresent",
            "mainClass": args.mainClass,
            # '-' is a special charter in YAML, use '\' to escape, need to remove
            "arguments": args.arguments.strip('\\').split(),
            "mainApplicationFile": args.mainApplicationFile,
            "sparkConf": {"spark.sql.avro.compression.codec": "deflate"},
            "sparkVersion": args.sparkVersion,
            "restartPolicy": {"type": args.restartPolicy},
        },
    }
    if args.driverSpec:
        inst["spec"]["driver"] = args.driverSpec
    if args.executorSpec:
        inst["spec"]["executor"] = args.executorSpec

    create_response = sparkapp.create(inst)

    expected_conditions = ["COMPLETED", "FAILED"]
    sparkapp.wait_for_condition(
        args.namespace, args.name, expected_conditions,
        timeout=datetime.timedelta(minutes=args.sparkApplicationTimeoutMinutes))
    if args.deleteAfterDone:
        sparkapp.delete(args.name, args.namespace)

if __name__== "__main__":
    main()
