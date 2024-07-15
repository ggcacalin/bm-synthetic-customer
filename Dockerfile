FROM python:3.10

WORKDIR /syncus

COPY . /syncus

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "./src/gpt_4o_agent.py"]