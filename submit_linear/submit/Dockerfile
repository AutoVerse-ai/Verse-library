FROM python:3.8
MAINTAINER Daniel Zhuang EMAIL dzhuang6@illinois.edu
RUN git clone https://github.com/AutoVerse-ai/Verse-library.git Verse-Library
RUN cd Verse-Library && git checkout arch2023
RUN cd Verse-Library
WORKDIR "/Verse-Library"
RUN pip install -r requirements.txt
RUN python -m pip install -e .
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN pip install treelib
