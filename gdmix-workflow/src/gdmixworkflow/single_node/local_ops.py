from subprocess import Popen, PIPE


def get_param_list(params):
    """ transform params from dict to list.
    """
    if isinstance(params, dict):
        for k, v in params.items():
            yield str(k)
            yield str(v)
    else:
        raise ValueError("job params can only be dict")


def get_tfjob_cmd(params):
    """ get tfjob command for local execution
    """
    cmd = ['python', '-m', 'gdmix.gdmix']
    for param in params:
        for k, v in param.items():
            if v != "" and v is not None:
                cmd.append(f"--{k}={v}")
    return cmd


def get_sparkjob_cmd(class_name, params, jar='gdmix-data-all_2.11.jar'):
    """ get spark command for local execution
    """
    cmd = ['spark-submit',
           '--class', class_name,
           '--master', 'local[*]',
           '--num-executors', '1',
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
    print(out.decode("utf-8"))
    if process.returncode:
        raise RuntimeError(f"ERROR in executing command: {str(' '.join(cmd))}\n\nError message:\n{err.decode('utf-8')}")
    else:
        print(err.decode("utf-8"))
