FROM linkedin/gdmix-dev

USER root

# Set default shell to /bin/bash
SHELL ["/bin/bash", "-cu"]

ENV GDMIX_VERSION=0.3.0

# Install GDMix trainer and workflow packages
RUN pip install gdmix-trainer==${GDMIX_VERSION} \
                gdmix-workflow==${GDMIX_VERSION} \
                pandas \
                notebook \
                jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install

WORKDIR /workspace/notebook

# Download gdmix-data jar for intermediate data processing
RUN wget -c https://linkedin.bintray.com/maven/com/linkedin/gdmix/gdmix-data-all_2.11/${GDMIX_VERSION}/gdmix-data-all_2.11-${GDMIX_VERSION}.jar -O gdmix-data-all_2.11.jar

# Add gdmix config examples
ADD *.config ./

RUN rm -rf ~/.gradle/caches/* ~/.cache/pip/* /var/cache/yum/* && yum clean all
