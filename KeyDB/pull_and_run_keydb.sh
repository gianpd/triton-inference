#/bin/bash!

CONTAINER_NAME='NJKeyDB-server'

if [[ -z $1 ]];
then
   echo "No container name argument passed. Setting CONTAINER_NAME to $CONTAINER_NAME"
else
   echo "Setting CONTAINER_NAME to $1"
   CONTAINER_NAME=$1
fi

echo "pulling image eqalpha/keydb ..."
docker pull eqalpha/keydb
echo "running container ${CONTAINER_NAME} ..."
docker run -p 6379:6379 --name ${CONTAINER_NAME} -d eqalpha/keydb keydb-server /etc/keydb/keydb.conf #--requirepass mypassword 