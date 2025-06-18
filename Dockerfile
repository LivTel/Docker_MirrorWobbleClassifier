FROM python:3

#####################

LABEL author="RJS <r.j.smith@ljmu.ac.uk>"
LABEL description="Environmentin which to run vgg16 classifier for wild M1 spasms detection"

# Update base OS 
RUN apt -y update
RUN apt -y upgrade
# "--root-user-action ignore" suppresses warning that says not to run pip as root
RUN pip install --root-user-action ignore --upgrade pip

# Add temporary download space
#ENV DDIR=$WORKDIR/downloads
#RUN mkdir -p $DDIR

# Add the working directory
ENV WORKDIR=/app
#ADD . $WORKDIR			# Do not do this so early? Messes up cacheing?
WORKDIR $WORKDIR

# Where compose will mount the external persistent storage
RUN mkdir /mnt/external


# Install everything from requirements.txt
#COPY requirements.txt $WORKDIR
#RUN pip install --root-user-action ignore -r $WORKDIR/requirements.txt
# I prefer to pip them one at a time because when they are all wrapped in requirements.txt, Docker cannot cache images effectively
#os
#sys
#argparse
#socket
#timeit
# Maybe do not need <2. That was required for pylibtiff which I no longer use. Try newer numpy
RUN pip install --root-user-action ignore "numpy<2.0"
RUN pip install --root-user-action ignore astropy
RUN pip install --root-user-action ignore torch
RUN pip install --root-user-action ignore torchvision
RUN pip install --root-user-action ignore pillow


COPY app_ml_server.py  $WORKDIR


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
