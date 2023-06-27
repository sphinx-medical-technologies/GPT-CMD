#!/bin/sh
openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout selfsigned.key -out selfsigned.crt -subj "/C=AU/ST=/L=/O=Internet Widgits Pty Ltd/OU=/CN=/emailAddress="
#sudo cp gpt-server.service /etc/systemd/system/gpt-server.service
#sudo systemctl enable gpt-server
#sudo systemctl start gpt-server
#touch gpt.log
uvicorn server:app --host 0.0.0.0 --port 9002 --log-level info --ssl-keyfile selfsigned.key --ssl-certfile selfsigned.crt >> gpt.log 2>&1
