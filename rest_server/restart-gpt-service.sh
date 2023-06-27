#!/bin/sh
#sudo systemctl restart gpt-server
kill -9 $(cat gpt.pid)
uvicorn server:app --host 0.0.0.0 --port 9002 --log-level info --ssl-keyfile selfsigned.key --ssl-certfile selfsigned.crt
