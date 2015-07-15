[ -z "$1" ] && echo "Usage: $0 CONTAINER_ID" && exit 1
docker exec -ti $1 bash
