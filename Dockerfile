FROM python:3.10

WORKDIR /syncus

COPY . /syncus

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 80

ENV PYTHONPATH = "${PYTHONPATH}:/syncus/src"

CMD ["gunicorn", "-c", "gunicorn_config.py", "src.gpt_4o_agent:agent_app"]