#!/bin/sh

count=20
i=0
tot=0
# rm gen-measure.json

TEST_URL="http://0.0.0.0:9002"

while [ $i -lt $count ];do
	prompt="<START TRANSCRIPT>Date of birth  10-18-1967. Being referred to Dr. Mayer from Dr. Aralu.<END TRANSCRIPT>\n<START BIRTHDATE>"
	echo "prompt [$prompt]"
	response_time=$(curl -X "POST" -w "$i: %{time_total} %{http_code} %{size_download} %{url_effective}\n" -H "accept: application/json"  -H "Content-Type: application/json" -d "{ \"prompt\": [ \"$prompt\" ], \"bad_words\": [], \"temperature\": 0.8, \"top_k\": 50, \"top_p\": 1, \"min_length\": 0, \"max_length\": 20, \"penalty_alpha\": 0, \"repetition_penalty\": 1, \"early_stop\": false, \"end_sequence\": \"\", \"do_sample\": true, \"num_beams\": 1, \"return_prompt\": \"\", \"seed\": -1 }" -o gen-measure.json -s "$TEST_URL/generate/")
#	echo "response_time [$response_time]"
	val=$(echo $response_time | cut -f2 -d' ')
#	echo $val
	tot=$(echo "scale=3;${tot}+${val}" | bc)
#	echo $tot
	prompt=$(jshon -e prompt -e 0 < gen-measure.json | tr -d '"')
#	echo "prompt [$prompt]"
	gen_text=$(jshon -e gen_text -e 0 < gen-measure.json | tr -d '"')
	echo "gen_text [$gen_text]"
	last_string="<START BIRTHDATE>"
	after_last_string=${gen_text#*"$last_string"}
	response=$(echo $after_last_string | cut -d '<' -f 1 | tr -d '\\' | cut -d '.' -f 1)
	echo "response [$response]"
#	rm gen-measure.json
	i=$((i+1))
done

avg=$(echo "scale=3; ${tot}/${count}" |bc)
echo "   ........................."
echo "   AVG: $tot/$count = $avg"

sleep 30

