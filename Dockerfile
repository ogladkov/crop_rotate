FROM pytorch/pytorch

RUN apt update
RUN apt install -y git
RUN apt install wget
RUN apt install unzip
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN git clone https://github.com/ogladkov/crop_rotate.git
RUN pip install -r crop_rotate/requirements.txt
RUN pip install gdown
RUN gdown --id 1KqxXSmNq67IQPIxNFKyq0qY1WjbVLSzB
RUN gdown --id 18EM13etbQ5WB48dEqjfGa07mwQurURDj
RUN unzip images.zip
RUN mv images test
RUN rm images.zip

RUN python /workspace/crop_rotate/segmentation-infer.py
