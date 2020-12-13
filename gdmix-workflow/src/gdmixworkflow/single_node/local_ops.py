from subprocess import Popen, PIPE


def get_param_list(params):
    """ transform params from dict to list.
    """
    if isinstance(params, dict):
        for (k, v) in params.items():
            yield str(k)
            yield str(v)
    else:
        raise ValueError("job params can only be dict")


def get_tfjob_cmd(params):
    """ get tfjob command for local execution
    """
    cmd = ['python', '-m', 'gdmix.gdmix']
    for param in params:
        # Workaround for DetextArg until it's update with proper serialization override
        from detext.run_detext import DetextArg
        if type(param) is DetextArg:
            param = param._replace(feature_names=[','.join(param.feature_names)])
        cmd.extend(param.__to_argv__())
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
    returnCode = process.returncode
    print(out.decode("utf-8"))
    if returnCode != 0:
        raise RuntimeError(f"ERROR in executing commnd: {str(' '.join(cmd))}\n\nError message:\n{err.decode('utf-8')}")
    else:
        print(err.decode("utf-8"))
