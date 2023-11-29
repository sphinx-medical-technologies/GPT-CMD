# uvicorn server:app --host 0.0.0.0 --port 9002 --log-level debug

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from typing import Union
import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

f = open("gpt.pid", "w")
f.write(str(os.getpid()))
f.close()

from app import GptAPI
from app_typings import post_data
from config import Config

app = FastAPI()
config = Config()
app_runner = GptAPI(config)

@app.get('/')
def get_root():
    return {'message': 'This is the GPT-CMD API'}

@app.post("/generate/")
def generate(request: Request,post_data:post_data):
    return app_runner.generate(request,post_data)

@app.post("/reload_model/")
def reload_model(request: Request,post_data:post_data):
    return app_runner.reload_model(request,post_data)

@app.get("/status/")
def get_status(request: Request):
    return JSONResponse(status_code=200,content={"message":"GPT API is running"})

if __name__ == "__main__":
    print("Launch with uvicorn server:app --port 9002 --log-level debug")
