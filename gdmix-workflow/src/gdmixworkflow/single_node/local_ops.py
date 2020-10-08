from gdmixworkflow.common.constants import *
from subprocess import Popen,PIPE,STDOUT


def get_param_list(params):
    """ transform params from dict to list.
    """
    if isinstance(params, dict):
        kvList = [(str(k), str(v)) for (k, v) in params.items()]
        paramList = [elem for tupl in kvList for elem in tupl]
        return paramList
    else:
        raise ValueError("job params can only be dict")


def get_tfjob_cmd(params):
    """ get tfjob command for local execution
    """
    cmd = ['python', '-m', 'gdmix.gdmix']
    cmd.extend(get_param_list(params))
    return cmd


def get_sparkjob_cmd(class_name, params, jar='gdmix-data-all_2.11.jar'):
    """ get spark command for local execution
    """
    cmd = ['spark-submit',
           '--class', class_name,
           '--master', 'local[*]',
           '--num-executors','1',
           '--driver-memory', '1G',
           '--executor-memory', '1G',
           '--conf', 'spark.sql.avro.compression.codec=deflate',
           '--conf', 'spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs=false',
           jar]
    cmd.extend(get_param_list(params))
    return cmd


def run_cmd(cmd):
    """ run gdmix job locally.
    Params:
        cmd: shell command, e.g. ['spark-submit', '--class', ...]
    """
    process = Popen(cmd, stdout=PIPE, stderr=PIPE)
    # wait for the process to terminate
    out, err = process.communicate()
    returnCode = process.returncode
    print(out.decode("utf-8"))
    if returnCode != 0:
        raise RuntimeError("ERROR in executing commnd: {}\n\nError message:\n{}".format(
            str(' '.join(cmd)), err.decode("utf-8")))
    else:
        print(err.decode("utf-8"))
