#!/bin/sh

count=20
i=0
tot=0

while [ $i -lt $count ];do
	prompt="<START TRANSCRIPT>Date of birth  10-18-1967. Being referred to Dr. Mayer from Dr. Aralu.<END TRANSCRIPT>\n<START BIRTHDATE>"
	echo "prompt [$prompt]"
	response_time=$(curl -w "$i: %{time_total} %{http_code} %{size_download} %{url_effective}\n" -H "Content-Type: application/json" -d '{ "model": "/home/silvacarl/Desktop/models/gpt-cmd/", "prompt": "<START TRANSCRIPT>Date of birth  10-18-1967. Being referred to Dr. Mayer from Dr. Aralu.<END TRANSCRIPT>\n<START BIRTHDATE>", "max_tokens": 200 }' -o gen-measure.json -s "http://localhost:9002/v1/completions")
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

exit

export CUDA_VISIBLE_DEVICES=0 && python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9002 --model /home/silvacarl/Desktop/models/gpt-cmd

curl -X 'POST' 'http://216.153.49.118:9002/v1/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "model": "/home/silvacarl/Desktop/models/gpt-cmd", "prompt": [ "<START TRANSCRIPT>Date of birth  10-18-1967. Being referred to Dr. Mayer from Dr. Aralu.<END TRANSCRIPT>\n<START BIRTHDATE>" ], "max_tokens": 80, "temperature": 1, "top_p": 1, "n": 1, "stream": false, "logprobs": 0, "echo": false, "stop": "string", "presence_penalty": 0, "frequency_penalty": 0, "best_of": 5, "user": "gpt-j-6b", "top_k": -1, "ignore_eos": false, "use_beam_search": false }'

export CUDA_VISIBLE_DEVICES=1 && python -m vllm.entrypoints.openai.api_server --host 0.0.0.0 --port 9005 --model "EleutherAI/gpt-j-6b"

curl -X 'POST' 'http://216.153.49.118:9005/v1/completions' -H 'accept: application/json' -H 'Content-Type: application/json' -d '{
  "model": "EleutherAI/gpt-j-6b", "prompt": [ "what is metformin used for?" ], "max_tokens": 80, "temperature": 1, "top_p": 1, "n": 1, "stream": false, "logprobs": 0, "echo": false, "stop": "string", "presence_penalty": 0, "frequency_penalty": 0, "best_of": 5, "user": "gpt-j-6b", "top_k": -1, "ignore_eos": false, "use_beam_search": false }'

