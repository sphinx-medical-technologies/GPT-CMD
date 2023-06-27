#!/bin/sh

# Where the file post.txt holds the POST data:
# ab -p post.txt -H 'Content-Type: text/plain' -n 100 -c 1 http://aaa/bbb/message
# ab -p post.txt -T text/plain -n 100 -c 1 http://aaa/bbb/message
# ab -n 100 http://www.google.com/

# curl -X POST -d @file server:port -w %{time_connect}:%{time_starttransfer}:%{time_total}

# curl -X POST -d @file server:port --trace-time

curl_time() {
    curl -so /dev/null -w "\
   namelookup:  %{time_namelookup}s\n\
      connect:  %{time_connect}s\n\
   appconnect:  %{time_appconnect}s\n\
  pretransfer:  %{time_pretransfer}s\n\
     redirect:  %{time_redirect}s\n\
starttransfer:  %{time_starttransfer}s\n\
-------------------------\n\
        total:  %{time_total}s\n" "$@"
}

curl_time -X GET -H "Content-Type: application/json" "http://wordpress.com/"

count=5
i=0
tot=0
while [ $i -lt $count ];do
	res=$(curl -w "$i: %{time_total} %{http_code} %{size_download} %{url_effective}\n" -o "/dev/null" -s http://wordpress.com/)
	echo $res
	val=$(echo $res | cut -f2 -d' ')
	echo $val
	tot=$(echo "scale=3;${tot}+${val}" | bc)
	echo $tot
	i=$((i+1))
done

avg=$(echo "scale=3; ${tot}/${count}" |bc)
echo "   ........................."
echo "   AVG: $tot/$count = $avg"


