FROM ufoym/deepo:all-py36-jupyter
RUN apt-get -qq update && \
    apt-get install -y unzip \
                        zlib1g-dev \
                       libjpeg-dev \
                       zlib1g-dev \
                       libjpeg-dev \
                       xvfb \
                       libav-tools \
                       xorg-dev \
                       graphviz 
    
RUN pip install --upgrade tqdm \
                           graphviz \
                           nltk \
                           gensim \
                           bokeh \
                           editdistance \
			   jupyterlab \
                           seaborn && \
    python3 -m ipykernel.kernelspec
                 
COPY run_jupyter.sh /
CMD ["/run_jupyter.sh"]
