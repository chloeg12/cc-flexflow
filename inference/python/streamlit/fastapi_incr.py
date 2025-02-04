# Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
Running Instructions:
- To run this FastAPI application, make sure you have FastAPI and Uvicorn installed.
- Save this script as 'fastapi_incr.py'.
- Run the application using the command: `uvicorn fastapi_incr:app --reload --port PORT_NUMBER`
- The server will start on `http://localhost:PORT_NUMBER`. Use this base URL to make API requests.
- Go to `http://localhost:PORT_NUMBER/docs` for API documentation.
"""


from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
import flexflow.serve as ff
from flexflow.core import *
from flexflow.type import (
    OptimizerType
)
import uvicorn
import json, os, argparse
from types import SimpleNamespace
from typing import Optional, List, Dict
import time
from huggingface_hub import hf_hub_download

# Initialize FastAPI application
app = FastAPI()

# Define the request model
class PromptRequest(BaseModel):
    prompt: str

# data models
class Message(BaseModel):
    role: str
    content: str


# class ChatCompletionRequest(BaseModel):
#     model: Optional[str] = "mock-gpt-model"
#     messages: List[Message]
#     max_tokens: Optional[int] = 512
#     temperature: Optional[float] = 0.1
#     stream: Optional[bool] = False

class ChatCompletionRequest(BaseModel):
    max_new_tokens: Optional[int] = 1024
    messages: List[Message]
    peft_model_id: Optional[str] = None  # Add optional PEFT model ID for adapter

# For finetuning request
class FinetuneRequest(BaseModel):
    token: str
    peft_model_id: str
    dataset_option: str
    dataset: Optional[List[str]] = None
    dataset_name: Optional[str] = None
    lora_rank: int = 16
    lora_alpha: int = 16
    target_modules: Optional[List[str]] = ["down_proj"]
    learning_rate: float = 1e-5
    optimizer_type: str
    momentum: float
    weight_decay: float
    nesterov: bool = False
    max_steps: int = 2
    # max_steps: int = 10000

# Global variable to store the LLM model
llm = None

OPTIMIZER_TYPE_MAP = {
    "SGD": OptimizerType.OPTIMIZER_TYPE_SGD,
    "Adam": OptimizerType.OPTIMIZER_TYPE_ADAM,
}

# Registry to store CFFI objects mapped to string IDs
adapter_registry = {}

def get_configs():

    # Fetch configuration file path from environment variable
    config_file = os.getenv("CONFIG_FILE", "")

    # Load configs from JSON file (if specified)
    if config_file:
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Config file {config_file} not found.")
        try:
            with open(config_file) as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print("JSON format error:")
            print(e)
    else:
        # Define sample configs
        ff_init_configs = {
            # required parameters
            # "num_gpus": 8,
            "num_gpus": 4,
	        "memory_per_gpu": 20000,
            "zero_copy_memory_per_node": 40000,
            # optional parameters
            "num_cpus": 4,
            "legion_utility_processors": 8,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 4,
            "pipeline_parallelism_degree": 1,
            "offload": False,
            "offload_reserve_space_size": 8 * 1024, # 8GB
            "use_4bit_quantization": False,
            "use_8bit_quantization": False,
            # "enable_peft": False,
            "enable_peft": True,
            "peft_activation_reserve_space_size": 1024, # 1GB
            "profiling": False,
            "benchmarking": False,
            "inference_debugging": False,
            "fusion": True,
        }
        llm_configs = {
            # required parameters
            # "llm_model": "meta-llama/Llama-3.1-8B-Instruct",
            "llm_model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
            # optional parameters
            "cache_path": os.environ.get("FF_CACHE_PATH", ""),
            "refresh_cache": False,
            "full_precision": False,
            "prompt": "",
            "output_file": "",
        }
        # Merge dictionaries
        ff_init_configs.update(llm_configs)
        return ff_init_configs
    

# Initialize model on startup
@app.on_event("startup")
async def startup_event():
    global llm

    # Initialize your LLM model configuration here
    configs_dict = get_configs()
    configs = SimpleNamespace(**configs_dict)
    ff.init(configs_dict)

    ff_data_type = (
        ff.DataType.DT_FLOAT if configs.full_precision else ff.DataType.DT_HALF
    )
    llm = ff.LLM(
        configs.llm_model,
        data_type=ff_data_type,
        cache_path=configs.cache_path,
        refresh_cache=configs.refresh_cache,
        output_file=configs.output_file,
    )

    generation_config = ff.GenerationConfig(
        do_sample=False, temperature=0.9, topp=0.8, topk=1
    )
    llm.compile(
        generation_config,
        max_requests_per_batch=16,
        max_seq_length=2048,
        max_tokens_per_batch=1024,
        enable_peft_finetuning=True,
    )
    llm.start_server()

# API endpoint to generate response
@app.post("/generate/")
async def generate(prompt_request: PromptRequest):
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")
    
    # Call the model to generate a response
    full_output = llm.generate([prompt_request.prompt])[0].output_text.decode('utf-8')
    
    # Separate the prompt and response
    split_output = full_output.split('\n', 1)
    if len(split_output) > 1:
        response_text = split_output[1] 
    else:
        response_text = "" 
        
    # Return the prompt and the response in JSON format
    return {
        "prompt": prompt_request.prompt,
        "response": response_text
    }

@app.post("/register_adapter/")
async def register_adapter(peft_model_name: str = Query(..., description="PEFT model name")):
    """
    Attempt to register a LoRA adapter for inference.
    Download weights and validate the base model if not already done.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")

    # try:
    if not peft_model_name:
        raise ValueError("PEFT model name is required.")
    
    # Validate and download the LoRA adapter weights if needed
    llm.download_peft_adapter_if_needed(peft_model_name)

    # Create LoRA inference configuration
    lora_inference_config = ff.LoraLinearConfig(
        llm.cache_path,
        peft_model_name.lower(), # convert the peft_model_name to lowercase 
        base_model_name_or_path=llm.model_name,
    )

    # Try to get the peft model id
    try:
        # If the adapter is already registered, retrieve its ID
        peft_model_id = llm.get_ff_peft_id(lora_inference_config)
        print(f"Adapter {peft_model_name} already registered with ID {peft_model_id}.")

    except ValueError as e:
        # If the adapter is not registered, register it and then retrieve its ID
        print(f"Adapter {peft_model_name} not registered. Registering now...")
        llm.register_peft_adapter(lora_inference_config)
        peft_model_id = llm.get_ff_peft_id(lora_inference_config)

    # Save the mapping of PEFT model name to CFFI object
    adapter_registry[peft_model_name] = peft_model_id

    # Return the adapter ID and success status
    return {"peft_model_id": str(peft_model_id), "status": "success"}

    # except Exception as e:
    #     return {"status": "error", "detail": str(e)}


@app.post("/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    Generate a response, optionally using a registered LoRA adapter.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")

    print("received request:", request)

    try:
        # Use the PEFT adapter if specified
        if request.peft_model_id:
            # Retrieve the CFFI object using the PEFT model name
            peft_model_cffi = adapter_registry.get(request.peft_model_id)
            prompt = llm._LLM__chat2prompt([message.dict() for message in request.messages])

            request = Request(
                ff.RequestType.REQ_INFERENCE,
                prompt=prompt,
                max_new_tokens=request.max_new_tokens,
                peft_model_id=peft_model_cffi,
            )
            result = llm.generate(request)[0].output_text.decode("utf-8")
        else:            
            result = llm.generate(
                [message.dict() for message in request.messages],
                max_new_tokens=request.max_new_tokens,
            )[0].output_text.decode("utf-8")

        print("Returning response:", result)
        return {"response": result}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to generate response: {str(e)}"
        )

    return {
        "id": "1337",
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [{"message": Message(role="assistant", content=resp_content)}],
    }

@app.post("/finetuning/")
async def finetune(request: FinetuneRequest):
    """
    Endpoint to start LoRA finetuning based on the provided parameters.
    """
    if llm is None:
        raise HTTPException(status_code=503, detail="LLM model is not initialized.")

    print("received request:", request)
    # llm.download_peft_adapter_if_needed("goliaro/llama-3-8b-lora")
    # request.peft_model_id = "goliaro/llama-3-8b-lora"

    # llm.download_peft_adapter_if_needed("DreamGallery/task-14-meta-llama-Meta-Llama-3.1-8B-Instruct")
    request.peft_model_id = "DreamGallery/task-14-meta-llama-Meta-Llama-3.1-8B-Instruct"
    # try:
    if request.optimizer_type not in OPTIMIZER_TYPE_MAP:
        raise ValueError(f"Unsupported optimizer type: {request.optimizer_type}")

    optimizer_type = OPTIMIZER_TYPE_MAP[request.optimizer_type]

    # Prepare LoRA configuration for finetuning
    lora_finetuning_config = ff.LoraLinearConfig(
        llm.cache_path,
        request.peft_model_id.lower(),
        trainable=True,
        # init_lora_weights=False,
        init_lora_weights=True,
        base_model_name_or_path=llm.model_name,
        optimizer_type=optimizer_type,
        target_modules=request.target_modules,
        optimizer_kwargs={
            "learning_rate": request.learning_rate,
            "momentum": request.momentum,
            "weight_decay": request.weight_decay,
            "nesterov": request.nesterov,
        },
    )

    # Register the LoRA configuration with the model
    llm.register_peft_adapter(lora_finetuning_config)

    # Load the dataset
    file_path = None
    if request.dataset_option == "Upload JSON":
        dataset_dir = os.path.join(os.path.expanduser(llm.cache_path), "datasets", "uploaded")
        os.makedirs(dataset_dir, exist_ok=True)

        file_path = os.path.join(dataset_dir, "dataset.json")
        with open(file_path, "w") as f:
            json.dump(request.dataset, f)
    elif request.dataset_option == "Hugging Face Dataset":
        dataset_dir = os.path.join(os.path.expanduser(llm.cache_path), "datasets", "huggingface")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # try:
            # Load the dataset from Hugging Face
        from datasets import load_dataset

        print(f"Attempting to load dataset: {request.dataset_name}")

        # try:
            # Try loading the file without a token
        file_path = hf_hub_download(
            repo_id="SylvanL/Traditional-Chinese-Medicine-Dataset-SFT",
            filename="SFT_medicalKnowledge_source2_99334.json",
            repo_type="dataset",
            local_dir=dataset_dir,
        )

        # except Exception as e:
        #     # If an exception is raised, try again with the token
        #     print(f"Failed to load dataset without token. Error: {str(e)}")
        #     try:
        #         # dataset = load_dataset(
        #         #     request.dataset_name,
        #         #     use_auth_token=request.token,
        #         #     # cache_dir=dataset_filepath,
        #         # )
        #         file_path = hf_hub_download(
        #             repo_id=request.dataset_name,
        #             filename="SFT_medicalKnowledge_source2_99334.json",
        #             repo_type="dataset",
        #             use_auth_token=request.token
        #         )
        #     except Exception as token_error:
        #         raise HTTPException(
        #             status_code=400,
        #             detail=(
        #                 f"Failed to access the dataset '{request.dataset_name}'. "
        #                 f"Either the dataset does not exist, or you do not have access permissions with the provided token. Error: {str(e)}"
        #             ),
        #         )

    print(f"Dataset saved to {file_path}")

    # Create finetuning request
    finetuning_request = ff.Request(
        ff.RequestType.REQ_FINETUNING,
        peft_model_id=llm.get_ff_peft_id(lora_finetuning_config),
        dataset_filepath=file_path,
        max_training_steps=request.max_steps,
    )

    print("-----start generate-----")

    results = llm.generate(finetuning_request)[0].output_text.decode("utf-8")

    print("Returning response:", result)
    return {"response": result}

    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Finetuning failed: {str(e)}")
    

# Shutdown event to stop the model server
@app.on_event("shutdown")
async def shutdown_event():
    global llm
    if llm is not None:
        llm.stop_server()

# Main function to run Uvicorn server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Running within the entrypoint folder:
# uvicorn fastapi_incr:app --reload --port

# Running within the python folder:
# uvicorn entrypoint.fastapi_incr:app --reload --port 3000