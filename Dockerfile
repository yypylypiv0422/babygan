FROM webtunixdc/babydocker
RUN apt update
EXPOSE 8080
CMD python3 main_new_updated.py
