FROM linkedin/gdmix-dev

# Install Notebook
RUN pip install notebook jupyter_contrib_nbextensions
RUN jupyter contrib nbextension install

ARG WORK_DIR="/workspace/notebook"
WORKDIR ${WORK_DIR}

# Install gdmix components
ARG BUILD_DIR="build_dir"
RUN mkdir ${BUILD_DIR}
COPY gdmix-trainer ${BUILD_DIR}/gdmix-trainer
COPY gdmix-workflow ${BUILD_DIR}/gdmix-workflow
COPY gdmix-data-all ${BUILD_DIR}/gdmix-data-all
COPY gdmix-data ${BUILD_DIR}/gdmix-data
COPY gradle ${BUILD_DIR}/gradle
COPY scripts ${BUILD_DIR}/scripts
COPY build.gradle ${BUILD_DIR}/
COPY settings.gradle ${BUILD_DIR}/
COPY gradlew ${BUILD_DIR}/

# Install GDMix components
RUN cd ${BUILD_DIR}
RUN python -m pip install --upgrade pip && pip install --upgrade setuptools pytest
RUN cd ${BUILD_DIR}/gdmix-trainer && pip install . && cd ../..
RUN cd ${BUILD_DIR}/gdmix-workflow && pip install . && cd ../..
RUN cd ${BUILD_DIR} && sh gradlew shadowJar && cp build/gdmix-data-all_2.11/libs/gdmix-data-all_2.11*.jar  ${WORK_DIR}

# Download and process movieLens data 
RUN cp ${WORK_DIR}/${BUILD_DIR}/scripts/download_process_movieLens_data.py .
RUN pip install pandas
RUN python download_process_movieLens_data.py

# Copy gdmix configs for movieLens exmaple
RUN cp ${WORK_DIR}/${BUILD_DIR}/gdmix-workflow/examples/movielens-100k/*.yaml .

RUN rm -rf ~/.gradle/caches/* ~/.cache/pip/* ${WORK_DIR}/${BUILD_DIR}

