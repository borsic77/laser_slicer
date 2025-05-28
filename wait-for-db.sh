#!/bin/bash
# Usage: ./wait-for-db.sh host:port -- command args

set -e

hostport="$1"
shift

# Split host and port
IFS=':' read -r host port <<< "$hostport"

echo "Waiting for $host:$port to be available..."

while ! nc -z "$host" "$port"; do
  sleep 0.5
done

echo "$host:$port is available — executing command"
exec "$@"