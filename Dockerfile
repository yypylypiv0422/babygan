FROM webtunixdc/babydocker
RUN apt update
EXPOSE 8000
CMD python3 app.py
