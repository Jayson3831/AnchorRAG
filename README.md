# Towards Open-World Retrieval-Augmented Generation on Knowledge Graph: A Multi-Agent Collaboration Framework (WWW2026) 
<p align="center">
  <img src="framework.png" alt="AnchorRAG"/>
</p>

## News!
Our paper has been accepted for publication at WWW 2026! 

## Structure
+ `data/`: Datasets
+ `Freebase/`: Directory for storing the knowledge graph and its index files
+ `freebase_qa/`: Source code
+ `requirements.txt`: Project dependencies and environment configuration

## Knowledge Graph: Download and Preprocessing
All commands in this section should be executed from within the `Freebase/` directory.
1. Follow the data download and preprocessing steps outlined in the [Freebase Setup](https://github.com/GasolSun36/ToG/tree/main/Freebase)guide. Please ensure that the `virtuoso` service is running throughout the subsequent testing process. The service can be started in the background using the `../bin/virtuoso-t` command.
2. Run `filter_entities.py` to extract all entity names.
3. Run `build_index.py` to build the search index.

## Get started
1. Before execution, you must configure your Large Language Model (LLM) deployment by modifying the `run_llm` function in `llm_handler.py`. You can find the function at this [link](https://github.com/jxu3831/RAGE/blob/main/freebase_qa/core/llm_handler.py).
2. Execute the following commands to run the main script:
```
pip install -r requirements.txt
cd ../freebase_qa/
python main.py dataset webqsp --LLM gpt-4o-mini --openai_api_keys 'your_keys' --url 'your_llm_url' --engine 'azure_openai' --method rage
```
The `--LLM` argument is used for logging and naming result files. It does not change the model actually deployed by the `run_llm` function. To switch to a different LLM, you must modify both this argument and the implementation within `run_llm`.

## Parameter
We employed a consistent set of parameters for all experiments conducted on the datasets, as detailed below:
```
--width 3
--depth 3
--relation_num 5
--agent_count 3
```

## How to cite
If you are interested or inspired by this work, you can cite us by:
```sh

```