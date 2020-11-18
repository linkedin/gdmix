FROM centos:centos7

USER root

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ENV PYTHON_VERSION=3.7.4

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
    gcc \
    libffi-devel \
    make \
    sqlite-devel

ENV JAVA_HOME=/etc/alternatives/jre

# Install python 3.7.4
RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz
RUN tar xzf Python-${PYTHON_VERSION}.tgz && \
    cd Python-${PYTHON_VERSION} && \
    ./configure --enable-optimizations --enable-loadable-sqlite-extensions && \
    make -j16 altinstall
RUN cd .. && rm -rf Python*

# Make python3 the default python
RUN rm -rf /usr/bin/python && \
    ln -s /usr/local/bin/python3.7 /usr/bin/python && \
    ln -s /usr/local/bin/pip3.7 /usr/bin/pip

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

WORKDIR /workspace/notebook

# Download and process movieLens data
ADD download_process_movieLens_data.py .
RUN pip install numpy \
                pandas \
                tensorflow==1.15.2 \
                pytest
RUN python download_process_movieLens_data.py

RUN rm -rf ~/.gradle/caches/* ~/.cache/pip/* /var/cache/yum/* && yum clean all
