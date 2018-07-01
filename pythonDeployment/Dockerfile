FROM python:3

ADD model.py /

ADD hello.py /

RUN pip install pystrich

RUN pip install flask

EXPOSE 5000

CMD [ "python", "./hello.py" ]