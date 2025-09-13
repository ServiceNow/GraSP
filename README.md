<div align="center">
  <img width=30% src="https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/grasp_logo.png">

  <h1>GRASP: GRAph-oriented Synthetic data generation Pipeline</h1>

<a href="https://github.com/ServiceNow/GraSP/releases">
    <img alt="Documentation" src="https://img.shields.io/github/v/release/ServiceNow/GraSP?logo=bookstack&logoColor=white"/></a>
<a href="https://servicenow.github.io/GraSP">
    <img alt="Documentation" src="https://img.shields.io/badge/MkDocs-Documentation-green.svg"/></a>
<a href="http://arxiv.org/abs/2508.15432">
    <img src="https://img.shields.io/badge/arXiv-2508.15432-B31B1B.svg" alt="arXiv"></a>
<a href="LICENSE">
    <img alt="Licence" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"/></a>

<br>
</div>


Framework to easily generate complex synthetic data pipelines by visualizing and configuring the pipeline as a
computational graph. [langgraph](https://python.langchain.com/docs/langgraph/) is used as the underlying graph
configuration/execution library. Refer
to [langraph examples](https://github.com/langchain-ai/langgraph/tree/main/examples) to get a sense of the different
kinds of computational graph which can be configured.
<br>
<be>

## Introduction

GraSP Framework is created to generate synthetic data. As it is a complex process to define the flow, this design simplifies the synthetic data generation process. GraSP platform will support the following:
- Defining the seed data configuration
- Define a task, which involves graph node configuration, flow between nodes and conditions between the node
- Define the output location to dump the generated data

Seed data can be pulled from either Huggingface or file system. Once the seed data is loaded, GraSP platform allows datagen users to write any data processing using the data transformation module. When the data is ready, users can define the data flow with various types of nodes. A node can also be a subgraph defined in another yaml file.

Each node can be defined with preprocessing, post processing, and LLM prompt with model parameters. Prompts can use seed data as python template keys.  
Edges define the flow between nodes, which can be conditional or non-conditional, with support for parallel and one-to-many flows.

At the end, generated data is collected in the graph state for a specific record, processed further to generate the final dictionary to be written to the configured data sink.

![GraspFramework](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/grasp_architecture.png)

---

## Components
GraSP architecture consists of various components, below are the four major component diagrams explains various modules.

### Data Handler
Data handler is used for reading and writing the data. Currently, it supports file handler with various file types and huggingface handler.
When reading data from huggingface, it can read the whole dataset and process, or it can stream chunk of data.

![DataHandler](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/component_data_handler.png)

### Graph Node Module
This module is responsible for building various kind of nodes like LLM node, Multi-LLM node, Lambda node, Agent node etc.
Each node is defined for various task, for example multi-llm node is used to load-balance the data among various inference point running same model.

![Nodes](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/component_nodes.png)

### Graph Edge Connection
Once node are built, we can connect them with simple edge or conditional edge.
Conditional edge uses python code to decide the path. Conditional edge helps implimenting if-else flow as well as loops in the graph.

![Edges](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/component_edges.png)

### Model clients
GraSP doesn't support inference within the framework, but it supports various clients, which helps connecting with different kind of servers.
For example, openai client is being supported by Huggingface TGI, vLLM server and Azure services. However, model configuration does not allow to change clients, but it can be configured in models code.

![ModelClient](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/component_model_client.png)

## Task Components

GraSP supports extendability and ease of implementation—most tasks are defined as graph configuration YAML files. Each task consists of two major components: a graph configuration and Python code to define conditions and processors.
YAML contains various parts:
* Data configuration : Configure file or huggingface as source and sink for the task.
* Data transformation : Configuration to transform the data into the format it can be used in the graph.
* Node configuration : Configure nodes and corresponding properties, preprocessor and post processor.
* Edge configuration : Connect the nodes configured above with or without conditions. 
* Output configuration : Configuration for data tranformation before writing the data into sink.

A node is defined by the node module, supporting types like LLM call, multiple LLM call, lambda node, and sampler node.  

LLM-based nodes require a model configured in `models.yaml` and runtime parameters. Sampler nodes pick random samples from static YAML lists. For custom node types, you can implement new nodes in the platform.

As of now, LLM inference is supported for TGI, vLLM, Azure, Azure OpenAI, Ollama and Triton compatible servers. Model deployment is external and configured in `models.yaml`.

<!-- ![GraspComponents](https://raw.githubusercontent.com/ServiceNow/GraSP/refs/heads/main/docs/resources/images/grasp_usecase2framework.png) -->


## Contact

To contact us, join our channel or send us an [email](mailto:grasp_team@servicenow.com)!

## License

The package is licensed by ServiceNow, Inc. under the Apache 2.0 license. See [LICENSE](LICENSE) for more details.

---

**Questions?**  
Open an issue or discussion! Contributions are welcome.
