FROM python:3

RUN pip install pystrich

RUN pip install flask

RUN pip3 install --upgrade \
    numpy \
    scipy \
    scikit-learn \
    matplotlib \
    pandas

EXPOSE 5000

ADD model.py /

ADD hello.py /

ADD regression_model.py /

RUN python ./regression_model.py

CMD [ "python", "./hello.py" ]