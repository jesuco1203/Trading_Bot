FROM python:3.12-slim
WORKDIR /app
COPY monitoring ./monitoring
COPY scripts ./scripts
COPY configs ./configs
RUN mkdir -p /app/data
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app PORT=8080
EXPOSE 8080
CMD ["python", "-m", "monitoring.web_app"]
