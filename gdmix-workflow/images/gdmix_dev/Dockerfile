FROM python:3.7

# Install spark 2.4
ARG spark_version=2.4.8
ARG spark_pkg=spark-${spark_version}-bin-hadoop2.7

RUN apt-get update
RUN apt-get install software-properties-common -y
RUN apt-add-repository 'deb http://security.debian.org/debian-security stretch/updates main'
RUN apt-get update && apt-get install openjdk-8-jdk git -y
RUN mkdir -p /opt/spark
RUN wget https://downloads.apache.org/spark/spark-${spark_version}/${spark_pkg}.tgz && tar -xf ${spark_pkg}.tgz && \
    mv ${spark_pkg}/jars /opt/spark && \
    mv ${spark_pkg}/bin /opt/spark && \
    mv ${spark_pkg}/sbin /opt/spark && \
    mv ${spark_pkg}/examples /opt/spark && \
    mv ${spark_pkg}/data /opt/spark && \
    mv ${spark_pkg}/kubernetes/tests /opt/spark && \
    mv ${spark_pkg}/kubernetes/dockerfiles/spark/entrypoint.sh /opt/ && \
    mkdir -p /opt/spark/conf && \
    cp ${spark_pkg}/conf/log4j.properties.template /opt/spark/conf/log4j.properties && \
    sed -i 's/INFO/ERROR/g' /opt/spark/conf/log4j.properties && \
    chmod +x /opt/*.sh && \
    rm -rf spark-*

ENV SPARK_HOME=/opt/spark
ENV PATH=/opt/spark/bin:$PATH
ENV SPARK_CLASSPATH=$SPARK_CLASSPATH:/opt/spark/jars/

RUN rm -rf ~/.gradle/caches/* ~/.cache/pip/*

