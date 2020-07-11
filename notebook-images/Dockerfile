FROM quay.io/odh-jupyterhub/s2i-spark-minimal-notebook:3.6

USER root

RUN pip install boto3 boto
RUN pip install pandas sklearn ipywidgets ipython_blocking jaydebeapi

# COPY ./source/anomaly-detection/. /opt/app-root/src/

# RUN mkdir -p /opt/app-root/src/.local/share/jupyter/runtime && chmod -R 777 /opt/app-root/src/

# RUN jupyter trust *.ipynb

# RUN chmod -R 777 /opt/app-root/src/.local/share/jupyter/notebook_secret

# RUN chmod -R 777 /opt/app-root/src/.pki/nssdb
# RUN chmod -R 777 /opt/app-root/src/.local/share/jupyter/nbsignatures.db

USER 10001
