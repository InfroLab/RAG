# Prerequisites.
This project is only runnable on Ubuntu or Windows with WSL. This project utilizes Nvidia Container Toolkit, Docker, TensorRT-LLM, ONNX-Runtime and more.

All dependencies are installed and deployed with docker.

# Prepare your system.
Run following script to install Nvidia Container Toolkit in your Linux environment.
> bash install-cuda-container-toolkit.sh

It will also install docker. If you experienced any docker related errors on later steps, there can be issues with installation of it on this step.

# Build project 
Project is defined mainly in 4 files: a docker-compose.yml and 3 Dockerfiles for each container. 

***semantic-search*** represents a container that performs a semnantic search over a FAISS Vector Index using `llama_index` package and a custom class `OptimumEmbedding` to do Indexing and Searching.

***llm*** TensorRT environment with a TensorRT-LLM built engine to run `Mistran-7b-Instruct-v0.2-int4`. All conversion to TRT engine are performed inside the container during build time.

***api*** API service to perform basic answering for prompts including prompts history. JSON-strings that correpospond to the following `pydantic`-format are expected:
```python
class Query(BaseModel):
    query: str
    history: Optional[List[Dict[str,str]]]
```

To build the project run:
> docker compose build

Installation process takes about 15-20 minutes. So, take a coffee break. All container will be equipped with `conda` environments with `Python 3.10.14` and a dependencies defined in corresponding `environment.yml`'s.

# Running the containers
> docker-compose up

***api*** container will wait for ***semantic-search*** and ***llm***, they willa also have GPU capabilities.