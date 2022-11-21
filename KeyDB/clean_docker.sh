docker rm --force $(docker ps -aq)
docker rmi $(docker images -aq)
docker volume rm $(docker volume ls -qf dangling=true)
