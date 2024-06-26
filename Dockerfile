FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt .
RUN  pip3 install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
COPY disc_questions.csv .
COPY app.py ${LAMBDA_TASK_ROOT}
CMD [ "app.handler" ]


