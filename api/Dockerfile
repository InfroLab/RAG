FROM nvcr.io/nvidia/pytorch:20.12-py3
WORKDIR /app
COPY ./llm /app
RUN echo -e "yes\n" | bash /app/install-deps.sh
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]