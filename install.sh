# !/bin/bash
apt update
# TODO: Automatically set Geographic Area
apt install r-base 
# ENV TZ=Europe/Madrid
# RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# TODO: Automatically install R packages
# R
# install.packages("ks")
pip install -r requirements.txt
# TODO: Modify tikzplotlib
