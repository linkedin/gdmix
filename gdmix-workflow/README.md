# GDMix Workflow
GDMix-workflow is a workflow generation toolkit to orchestrate training jobs of the [GDMix](https://github.com/linkedin/gdmix), which is a framework to train non-linear fixed effect and random effect models. By providing a [GDMix config](gdmix_config.md), GDMix-workflow can generate jobs and run them directly, or generate a YAML file that can be uploaded to Kubeflow Pipeline to run training job distributedly on Kubernetes cluster with Kubeflow Pipeline deployed.

## Configuration
GDMix-workflow supports two modes, single_node and distributed. For single_node mode, user will need to install the [gdmix-workflow](https://pypi.org/project/gdmix-workflow/) package and spark, GDMix-workflow will prepare jobs and run them on the node. For distributed mode, GDMix-workflow generates a YAML file that can be uploaded to Kubeflow Pipeline. We'll explain more about distributed mode in the section [Run on Kubernetes](#Run-on-Kubernetes).
Once the `gdmix-workflow` package is installed (`pip install gdmix-workflow`),  user can call
```
python -m gdmixworkflow.main
```
plus following parameters:
  - **--config_path**: path to gdmix config. Required.
  - **--mode**: distributed or single_node. Required.
  - **--jar_path**: path to the gdmix-data jar for GDMix processing intermediate data. Required by single_node mode only.
  - **--workflow_name**: name for the generated zip file to upload to Kubeflow Pipeline. Required by distributed mode only.
  - **--namespace**: Kubernetes namespace. Required by distributed mode only.
  - **--secret_name**: secret name to access storage. Required by distributed mode only.
  - **--image**: image used to launch gdmix jobs on Kubernetes. Required by distributed mode only.
  - **--service_account**: service account for the `spark-on-k8s-operator` to launch spark job. Required by distributed mode only.

## Run GDMix workflow on Kubernetes for distributed training
GDMix's distributed training is based on [Kubernetes](https://kubernetes.io/docs/home/), and leverages Kubernetes job scheduling services [Kubeflow](https://www.kubeflow.org/docs/started/getting-started/) and [spark-on-k8s-operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator) to run TensorFlow and Spark job distributedly on Kubernetes, and uses [Kubeflow Pipeline](https://www.kubeflow.org/docs/pipelines/overview/pipelines-overview/) to orchestrate jobs. Besides that, a centralized storage is needed for storing training data and models. User can use
[Kubernetes-HDFS](https://github.com/apache-spark-on-k8s/kubernetes-HDFS/tree/master/charts) or [NFS](https://www.kubeflow.org/docs/other-guides/kubeflow-on-multinode-cluster/#background-on-kubernetes-storage) as the centralized storage.

### Create a Kubernetes cluster, deploy required services
To run GDMix in the distributed mode, user needs to create a Kubernetes cluster, and deploy following services:

- [Kubeflow tf-operator](https://www.kubeflow.org/docs/components/training/tftraining/#deploy-kubeflow)
- [spark-on-k8s-operator](https://github.com/GoogleCloudPlatform/spark-on-k8s-operator#installation)
- [Kubeflow Pipeline](https://www.kubeflow.org/docs/pipelines/installation/overview/)
- [NFS Server Provisioner](https://github.com/helm/charts/tree/master/stable/nfs-server-provisioner)

### Generate task YAML file and upload to Kubeflow Pipeline UI
When the Kubernetes cluster and services are ready, with the provided GDMix config, GDMix-workflow can generate task YAML file that has job specification for the distributed TensorFlow and Spark jobs. User needs to upload it to Kubeflow Pipeline and start training.

## Run the [MovieLens](https://grouplens.org/datasets/movielens/) example
In this section we'll introduce how to train fixed effect and random effect models using GDMix for MovieLens data.
Please download and preprocess moveLens data to meet GDMix's need using the provided script [download_process_movieLens_data.py](../scripts/download_process_movieLens_data.py):
```
wget https://raw.githubusercontent.com/linkedin/gdmix/master/scripts/download_process_movieLens_data.py
pip install pandas
python download_process_movieLens_data.py
```
For distributed training, the processed movieLens data need to be copied to the centralized storage.

We'll also need a GDMix config, a reference of training logistic regression models for the fixed effect `global` and the random effects `per-user` and `per-movie` with distributed training can be found at [lr-distributed-movieLens.config](examples/movielens-100k/lr-distributed-movieLens.config).

### Run on single node
Please see the section [Try out the movieLens example](../README.md#Try-out-the-movieLens-example) in the root [README.md](../README.md) for details of how to run the movieLens example on single node.

### Run on Kubernetes for distributed training
To run on Kubernetes, as mentioned earlier, user will need to copy the processed movieLens data to the centralized storage, modify the input path fields such as `training_data_dir`,  `validation_data_dir`, `feature_file` and `metadata_file` of the GDMix config for distributed training [lr-distributed-movieLens.config](examples/movielens-100k/lr-distributed-movieLens.config).

If using the provided image [linkedin/gdmix](https://hub.docker.com/repository/docker/linkedin/gdmix), user can mount the processed movieLens data from the centralized storage to path `/workspace/notebook/movieLens` for each worker then no change is needed for the distributed training GDMix config [lr-distributed-movieLens.config](examples/movielens-100k/resources/lr-distributed-movieLens.config).


#### Generate YAML file
User will need to install the `GDMix-worklfow` package to generate the YAML file:
```
pip install gdmix-workflow
```

Download the example GDMix config for distributed training and generate the YAML file with following command:
```
wget https://raw.githubusercontent.com/linkedin/gdmix/master/gdmix-workflow/examples/movielens-100k/lr-distributed-movieLens.config

python -m gdmixworkflow.main --config_path lr-distributed-movieLens.config --mode=distributed --workflow_name=movieLens --namespace=default --secret_name default --image linkedin:gdmix --service_account default
```

Parameters of `namespace`, `secret_name` and `service_account` relate to the Kubernetes cluster setting and job scheduling operator deployments. A zip file named `movieLens.zip` is expected to be produced.

#### Upload to Kubeflow Pipeline
If the Kubeflow Pipeline is successfully deployed, use can forward the Pipeline UI to local browser, The command below forwards the Pipeline UI to the local port 9980:
```
kubectl -n default port-forward svc/ml-pipeline-ui 9980:80
```
Type `localhost:9980` in the local browser to view the Kubeflow Pipeline UI, upload the produced YAML file `movieLens.zip`(click button `Upload pipeline`), and then click button `Create run` to start the training. A snapshot of the movieLens workflow is shown below.

<figure>
  <p align="center"> <img src="../figures/gdmix-kubeflow-pipeline.png" alt="" />
  </br>
  <ficaption>GDMix Distributed Training on Kubeflow Pipeline</ficaption>
  </p>
</figure
