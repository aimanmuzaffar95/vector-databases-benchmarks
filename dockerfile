# Dockerfile
FROM pgvector/pgvector:pg16
# Optional: copy init scripts into the standard Postgres init directory
# (they run only on first DB initialization, i.e., when the data volume is empty)
COPY ./db/init/ /docker-entrypoint-initdb.d/