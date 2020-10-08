FROM centos:centos7

USER root

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

# Install required tools and libraries
RUN yum update -y
RUN yum  install -y \
    unzip \
    wget \
    which \
    git \
    java-1.8.0-openjdk \
    bzip2 \
    bzip2-devel \
    openssl-devel \
    jq \
    unzip \
    tree \
    curl \
    python3-devel \
    python3
ENV JAVA_HOME=/etc/alternatives/jre

RUN yum groupinstall -y 'Development Tools'

# Make python3 the default python
RUN rm -rf /usr/bin/python && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/pip3 /usr/bin/pip

# Yum uses python2, update its interpreter
RUN sed -i 's/python/python2/g' /usr/bin/yum
RUN sed -i 's/python/python2/g' /usr/libexec/urlgrabber-ext-down

# Install spark 2.4.7 and its dependencies
RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm && \
    mkdir -p /opt/spark && \
    touch /opt/spark/RELEASE && \
    echo "auth required pam_wheel.so use_uid" >> /etc/pam.d/su && \
    chgrp root /etc/passwd && chmod ug+rw /etc/passwd

ARG spark_version=2.4.7
ARG spark_pkg=spark-${spark_version}-bin-hadoop2.7
ARG img_path=kubernetes/dockerfiles
ARG k8s_tests=kubernetes/tests

RUN wget https://downloads.apache.org/spark/spark-${spark_version}/${spark_pkg}.tgz && \
    tar -xf ${spark_pkg}.tgz && \
    mv ${spark_pkg}/jars /opt/spark && \
    mv ${spark_pkg}/bin /opt/spark && \
    mv ${spark_pkg}/sbin /opt/spark && \
    mv ${spark_pkg}/kubernetes/dockerfiles/spark/entrypoint.sh /opt/ && \
    mv ${spark_pkg}/examples /opt/spark && \
    mv ${spark_pkg}/kubernetes/tests /opt/spark && \
    mv ${spark_pkg}/data /opt/spark && \
    mkdir -p /opt/spark/conf && \
    cp ${spark_pkg}/conf/log4j.properties.template /opt/spark/conf/log4j.properties && \
    sed -i 's/INFO/ERROR/g' /opt/spark/conf/log4j.properties && \
    chmod +x /opt/*.sh && \
    rm -rf spark-*

ENV SPARK_HOME /opt/spark
ENV PATH /opt/spark/bin:$PATH
ENV SPARK_CLASSPATH=$SPARK_CLASSPATH:/opt/spark/jars/

# Install GDMix trainer and workflow packages
RUN pip install gdmix-trainer \
                gdmix-workflow \
                pandas \
                notebook \
                jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install

RUN rm -rf ~/.gradle/caches/* ~/.cache/pip/* /var/cache/yum/* && yum clean all

WORKDIR /workspace/notebook

# Download gdmix-data jar for intermediate data processing
RUN wget -c https://linkedin.bintray.com/maven/com/linkedin/gdmix/gdmix-data-all_2.11/0.2.0/gdmix-data-all_2.11-0.2.0.jar \
    -O gdmix-data-all_2.11.jar

# Download and process movieLens data
ADD download_process_movieLens_data.py .
RUN python download_process_movieLens_data.py

# Add gdmix config examples
ADD *.config ./
