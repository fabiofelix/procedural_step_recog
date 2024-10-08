FROM pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="America/New_York"

WORKDIR /src/app

# install opencv libraries
RUN apt-get update && apt-get install -y python3-opencv git && rm -rf /var/lib/apt/lists/*

# install ptgctl
RUN pip install git+https://github.com/VIDA-NYU/ptgctl.git

# install dependencies
ADD deploy/requirements.txt deploy/
RUN pip install -r deploy/requirements.txt

# install model source
ADD setup.py README.md ./
ADD step_recog/__init__.py step_recog/
ADD act_recog/__init__.py act_recog/
RUN pip install -e .
ADD . .

ADD deploy deploy

CMD ["deploy/main.py", "run"]
