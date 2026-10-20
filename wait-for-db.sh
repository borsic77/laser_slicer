#!/bin/bash
# Usage: ./wait-for-db.sh host:port -- command args

set -e

hostport="$1"
shift
if [ "${1:-}" = "--" ]; then
  shift
fi

host=$(echo "$hostport" | cut -d: -f1)
port=$(echo "$hostport" | cut -d: -f2)

echo "Waiting for $host:$port..."

retries=60
while ! nc -z "$host" "$port"; do
  retries=$((retries - 1))
  if [ "$retries" -le 0 ]; then
    echo "Error: Timed out waiting for $host:$port" >&2
    exit 1
  fi
  sleep 1
done

echo "$host:$port is available — starting app."
exec "$@"
