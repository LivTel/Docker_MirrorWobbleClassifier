FROM python:3

#####################

LABEL author="RJS <r.j.smith@ljmu.ac.uk>"
LABEL description="Environmentin which to run vgg16 classifier for wild M1 spasms detection"

# Add the working directory and temporary download space
ENV WORKDIR=/app
ENV DDIR=$WORKDIR/downloads
RUN mkdir -p $DDIR
RUN mkdir /mnt/external

# Update base OS 
RUN apt -y update
RUN apt -y upgrade

ADD . $WORKDIR
WORKDIR $WORKDIR

# "--root-user-action ignore" suppresses warning that says not to run pip as root
RUN pip install --root-user-action ignore --upgrade pip

# Install everything from requirements.txt
#COPY requirements.txt /usr/src/app/
RUN pip install --root-user-action ignore -r requirements.txt


# Install build tools
#RUN yum install -y gcc gcc-c++ kernel-devel wget git curl make tcsh \
#  && yum install -y python-devel libxslt-devel libffi-devel openssl-devel

# pillow needs zlib libjpeg python
#RUN yum -y install zlib zlib-devel libjpeg libjpeg-devel python-devel python3-devel


# Install Python package manager pip 
#RUN curl "https://bootstrap.pypa.io/get-pip.py" -o "$DDIR/get-pip.py" \
#  && python $DDIR/get-pip.py

# Install Python packages
# backports.functools-lru-cache is necessary to solve the "missing site-package/six" problem
# 
# Had a lot of trouble installing torch. I think the VM ran out of memory. Symptom was
# pip ended with messge "killed" and nothing more, no diagnostics or error message.
# Adding --no-cache-dir may help?
# VM just needs more RAM?
#
#RUN pip install backports.functools-lru-cache \


#RUN pip3 install --no-cache-dir numpy
#RUN pip3 install --no-cache-dir "numpy<2.0"

#RUN pip3 install --no-cache-dir astropy

#  && pip3 install --no-cache-dir argparse
#  && pip3 install --no-cache-dir future
#  && pip3 install --no-cache-dir torch
#  && pip3 install --no-cache-dir pillow
#  && pip3 install --no-cache-dir torchvision




# install dnf
#RUN yum install -y epel-release
#RUN yum install -y dnf

# Remove the epel-release after sorting out dnf
#RUN yum remove -y epel-release

# Add non-root user "data" inside the container
#RUN dnf install -y sudo \
#  && groupadd -g 600 web \
#  && adduser -u 501 -g 600 data \
#  && echo "user ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/user \
#  && chmod 0440 /etc/sudoers.d/user
