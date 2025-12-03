# Modified HeCBench for Roofline Analysis

We took this version of HeCBench and modified it to build the CUDA and OMP codes to gather their roofline performance data.
So far we have a large portion of the CUDA and OMP codes building without issue. We use CMake because the `autohecbench.py` was giving us trouble with easily switching out compilers and build options. There were also many issues with individual makefiles, so we decided to put all the build commands into one big `CMakeLists.txt` file for simplicity. We also wanted to create distinct phases of building and gathering data which wasn't too easy with `autohecbench.py`.

We might change our automated build and data gathering process in the future, for now what we have is working fine.

Target codes thus-far:
- CUDA (omitted MPI-based for now)
  - We are able to build 445/491 (90%) of the CUDA targets
  - We purposely skip building 46/491 due to MPI requirements, missing dependencies, or execution errors (mainly segfaults and out-of-memory errors)
- OMP
  - We are able to build 303/320 (94.6%) of the OMP targets
  - We purposely skip building 17/320 due to MPI requirements or missing build dependencies
  
## Citing our Work

**This work is a continuation of our previous work** that was presented at the [HPDC 2025](https://hpdc.sci.utah.edu/2025/) [AI4Sys Workshop](https://ai4sys.github.io/).
Upon having our paper accepted, we pre-published on arXiv in case people wanted to cite us.

📃📃 [paper link here](https://dl.acm.org/doi/10.1145/3731545.3743645) 🔗🔗

📜 BibTeX reference below.
```
@inproceedings{10.1145/3731545.3743645,
author = {Bolet, Gregory and Georgakoudis, Giorgis and Menon, Harshitha and Parasyris, Konstantinos and Hasabnis, Niranjan and Estes, Hayden and Cameron, Kirk and Oren, Gal},
title = {Can Large Language Models Predict Parallel Code Performance?},
year = {2025},
isbn = {9798400718694},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3731545.3743645},
doi = {10.1145/3731545.3743645},
booktitle = {Proceedings of the 34th International Symposium on High-Performance Parallel and Distributed Computing},
articleno = {41},
numpages = {6},
keywords = {roofline model, LLMs, CUDA, OpenMP, GPU, performance},
location = {University of Notre Dame Conference Facilities, Notre Dame, IN, USA},
series = {HPDC '25}
}
```

## Github Actions Status

For checking the build process, we include the following Github Action in the `.github/workflows/buildAllCodesGithubAction.yml` file.
This sets up a Docker container with all the necessary programs and code to build all the HecBench codes with our CMakeLists approach.
We also provide a `Dockerfile` so you can set up and run our code on your own system with a GPU.

[![Build ALL CUDA/OMP Codes](https://github.com/Scientific-Computing-Lab/gpuFLOPBench/actions/workflows/buildAllCodesGithubAction.yml/badge.svg?branch=main)](https://github.com/Scientific-Computing-Lab/gpuFLOPBench/actions/workflows/buildAllCodesGithubAction.yml)

## Docker Setup Instructions

For ease-of-reproducibility, we supply a `Dockerfile` with the necessary steps to recreate our environment and dataset using your own GPU hardware.
The following is a list of steps to help you get set up and into the main bash shell of the container.

‼️‼️
We note that the base container image will take up about 15 GB of storage space, which then jumps to 40 GB when we build the container; once we start building codes and gathering profiling data, the disk usage will jump up to about 50 GB.
Please ensure your system has enough storage space before continuing.
Additional note: we provide all the sampling and code-scraping data -- therefore you can simply use a non-NVIDIA-gpu-enabled machine to run the LLM queries with our dataset.
‼️‼️

### Container on NVIDIA GPU-Enabled Host

```
git clone https://github.com/Scientific-Computing-Lab/gpuFLOPBench.git ./gpu-flopbench

# we only really need the Dockerfile from the repo
cd ./gpu-flopbench

# this takes about 5-15 minutes
docker build --progress=plain -t 'gpu-flopbench' .

# please make sure that the Settings > Resources > Network > 'Enable Host Networking' option is enabled on Docker Desktop
# this is so you can run and view Jupyter Notebooks
docker run -ti --network=host --gpus all --name gpu-flopbench-container --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all gpu-flopbench

docker exec -it gpu-flopbench-container /bin/bash
```

### Container on Macbook (Apple Silicon M1/2/3/4) -- no NVIDIA GPU
```
git clone https://github.com/Scientific-Computing-Lab/gpuFLOPBench.git ./gpu-flopbench

# we only really need the Dockerfile from the repo
cd ./gpu-flopbench

# this takes about 2 minutes on my Macbook Air M4
docker build --platform=linux/amd64 --progress=plain -t 'gpu-flopbench' .

# please make sure that the Settings > Resources > Network > 'Enable Host Networking' option is enabled on Docker Desktop
docker run -ti --network=host --name gpu-flopbench-container --platform=linux/amd64 gpu-flopbench

docker exec -it gpu-flopbench-container /bin/bash
```

### Starting / Stopping Container

You can later start/stop the container using:
```
docker start gpu-flopbench-container
docker stop gpu-flopbench-container
```
The filechanges in the container will be persisted, unless you delete the container.

### Windows GPU-Enabled Docker Host -- Extra Steps

Note: if you're on a **Windows Docker Desktop** host, be sure to enable the following for GPU access:
```
NVIDIA Control Panel >> Desktop Tab >> Enable Developer Settings (make sure it's enabled)

then

NVIDIA Control Panel >> Select a Task... Tree Pane >> Developer (expand section) >> Manage GPU Performance Counters >> Allow access to the GPU performance counters to all users (make sure this is enabled)

then

restart Docker Desktop
```

# Steps to Run Scripts

Here we explain (at a high level) the steps used in the individual phases of data creation and collection for this work.
The later subsections of this README file explain each step in further detail.

1) **Source Code Scraping**: In this step we perform a simple naive concatenation of all the C/C++ files for each executable in the `gpu-flopbench/src` directory.
This same set of source codes was based off the [HeCBench Suite](https://github.com/zjin-lcf/HeCBench), where we take a focus on only the CUDA programs of the suite.
These scraped codes are used as inputs to the LLMs that we later query asking them to predict the FLOP counts of the target CUDA kernels.
In doing this scraping, we also record the command line input arguments to each executable that we run, passing these as input with the LLM query.

2) **Program Building and Kernel Extraction**: We took the HeCBench programs and built them all using a custom `CMakeLists.txt` file where we had fine control over the compiler and compilation process for each program.
We're able to build 445 of the programs which give us a reasonable amount of variety in the types of codes we sample.
Once all the programs are built, we execute them using the `ncu` NVIDIA profiler to record their performance counter data for each kernel we detect from the `cuobjdump` of the executables.
We only profile the first incovation of each kernel, as profiling all the invocations takes forever, and we really only want to predict on the first.

3) **Scraped Source Manual Annotation**: Given we had scraped the source codes, we wanted to differentiate the kernels that had FLOP counts which could be directly computed via static analysis (and constant propagation), and those which had implicit/indirect/hidden FLOP counts.
These hidden FLOP counts come from code attributes such as data-dependent warp divergence, math/intrinsic functions, external library calls, and floating point division. 
We consider these types of codes to be *hard* to predict on, as the LLMs would need to have runtime information which they are not given.
The codes that lack these properties, we consider them to be *easy*, where an LLM (or human expert) can directly calculate the number of FLOPs performed on the invocation of the target kernel.
We ended up creating a [Streamlit UI](https://github.com/streamlit/streamlit) to have human annotators go through each scraped source code and identify the code attributes used in each target kernel.
This allowed us to understand the code attributes/features that the LLMs struggled the most with predicting on.

4) **Dataset Creation**: Here we join the three previous steps, taking the scraped source code, performacne counter data, and manual annotations to form the dataset.
We emit two `csv` files, corresponding to the *easy* and *hard* datasets, respectively.
For convenience, we bundle these into a single CSV file `/gpu-flopbench/dataset-creation/gpuFLOPBench.csv`.

5) **LLM Querying**: For this step, we use the created dataset to ask the LLMs to predict the FLOP count of the scraped CUDA kernels.
We used [LangGraph](https://github.com/langchain-ai/langgraph) to make queries to the target models, where we store results in the `/gpu-flopbench/llm-querying/checkpoints` directory (about 2.3 GB of data).







## Note on Running Scripts

Our repo already provides all the intermediate files that would be generated during the following steps:

1) Source Code Scraping
2) Executable Building, Kernel Extraction, Kernel Profiling
3) Manual Code Annotations
4) Dataset Creation
5) LLM Querying

This means that you can run our same scripts to recreate results or add to the dataset, but if your system doesn't have a compatible NVIDIA GPU, you can at the very least see all our steps in deeper detail through this Docker container.
We heavily utilize Git LFS for large data files, so you can see all our raw LLM query results through the Langgraph SQLITE logging API.
Our LLM queries and their returned responses are stored in the `gpu-flopbench/llm-querying/checkpoints/paper-datasets/*.sqlite` files.

## Scraping Source Codes (`/gpu-flopbench/source-scraping`)

Before we can start gathering performance counter data, we need to do a source code scrape. 
This is done because once we start trying to build/run our codes, a lot of additional files will be getting unzipped that could accidentally pollute the contexts of the scraped codes.

```
## Output file already provided -- no need to run these commands

cd $GPU_FLOPBENCH_ROOT/source-scraping

python3 simpleScrapeKernels.py --skipSASS --cudaOnly --outfile="simple-scraped-kernels-CUDA-only.json"
```
This will generate a file called `simple-scraped-kernels-CUDA-only.json`, which is the full textual representation of each program with concatenated files.
This means that CUDA kernels that come from the same parent program have the same source code that gets passed to an LLM.


## Docker Data Collection Instructions -- CUDA program building & profiling (`/gpu-flopbench/cuda-profiling`)

Once you're in the main bash shell of the container, you should be by default in the `/gpu-flopbench` directory with a conda environment called `gpu-flopbench`. 
We can now start building all the codes and collecting their performance counter data! 🌈😊

Run the following commands from the `gpu-flopbench` main project directory within the Docker container (they should work without issue):
```
source ./runBuild.sh
```
^ Depending on the number of cores on your CPU, this can take anywhere from 5-20 minutes.
It's essentially building all the codes using our `CMakeLists.txt` file.
Once this is done, we can start gathering CUDA kernel profiling data with the following command:
```
cd $GPU_FLOPBENCH_ROOT/cuda-profiling

LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH DATAPATH=$PWD/../src/prna-cuda/data_tables SLU_PATH=$PWD/../src/slu-cuda/src python3 ./gatherData.py --outfile=profiling-data.csv 2>&1 | tee -a runlog.txt
```
^ This process will take about 10-15 hours, so please have someone around to babysit in case any unexpected issues arise.
We tested this on our own Docker container and had no issues, aside from timeouts for long-running codes or out-of-memory exceptions.






## Source Code Manual Categorization (`/gpu-flopbench/manual-code-classification`)
This part of the dataset is the human-annotated source codes (no need to run anything).
We manually inspected each of the scraped kernels and flagged the codes for the following properties:

1) Warp Divergence (i.e: conditional statements)
2) Calling a `__device__` function
3) Having recursion
4) Data-dependent Warp Divergence (i.e: conditional statements that depend on data read-in or calculated at runtime)
5) FLOP Division
6) Calls to external libraries or functions (e.g: CCCL/CUB, cuSparse, cuFFT)
7) Calling special math functions / intrinsics (e.g: sin/cos, `__ddiv_rd`)
8) Having common float subexpressions (i.e: repeated calculations)

If a code is flagged with any of the features 4-8, it is considered a *hard* code, that can NOT be directly statically analyzed, and thus could host hidden FLOP operations.
We provide this manually-annotated dataset in the file: `$GPU_FLOPBENCH_ROOT/manual-code-classification/manually_classified_CUDA_kernels.json`.
The UI for performing this classification was served through Streamlit, which allowed us to quickly get through manual kernel categorization efforts.
The file for launching the classification UI is: `$GPU_FLOPBENCH_ROOT/manual-code-classification/manualStreamlit_classify.py`, although we prodive our already-made classifications in the `manually_classified_CUDA_kernels.json` file.



### Automatic Scraping for Annotation
In order to manually classify each of the kernels, we needed a way to automatically focus on the CUDA kernel of interest for each profiled kernel.
We performed a step where we attempted to automatically scrape the CUDA kernel function definitions and their call path code, but given the complexity of C++, it was challenging.
The script below produces an output file called `extracted_CUDA_kernels.json` which is the result of these efforts.
```
## Output file already provided -- no need to run these commands

cd $GPU_FLOPBENCH_ROOT/source-scraping

python3 extract_kernels.py
```
The main problem with this approach is that we used TreeSitter for static concrete syntax tree (CST) parsing, which made it challenging to handle all the edge cases of C++, so lots of codes were not fully scraped (e.g: missing relevant callee code, includes code from CUDA headers, includes incorrect overloaded callee function definitions).
We end up using these scrapes in the Streamlit UI for manual annotations, where an annotator would often have to revert back to the full source code of a kernel to properly flag its features.

We hope to improve this process in the future, but for now it's mostly a human-based effort. 
Before this human-based approach, we did try to use LLMs, but they struggled to correctly extract code in many cases. Since we couldn't put much confiedence in LLMs to reliably perform this task for us, we scrapped the fully LLM-automated scraping approach.



# Dataset Creation (`/gpu-flopbench/dataset-creation`)
Now that we have the performance counter profiling data, scraped kernels, and manual annotations, we can create the dataset.
We break the process up into two steps: (1) creation of the *easy* subset and (2) creation of the *hard* subset, where we then join the two subsets into one dataset.
This step is reliant on Jupyter notebooks, so spin up the Jupyter server using the following command:
```
jupyter notebook --allow-root --no-browser --ip=0.0.0.0 --port=8888 --NotebookApp.token=''
```
Since we built the Docker container using the `--network=host` flag, you should be able to connect to the Jupyter server from your host machine browser at: [http://127.0.0.1:8888/tree/](http://127.0.0.1:8888/tree/)

## *Easy* Subset Creation
We make the *easy* subset using the notebook called `create_easy_dataset_to_query.ipynb`, it emits a file called `kernels_to_inference_balanced_with_compile_commands.csv`.
This file is used by the `/gpu-flopbench/llm-querying/dataset_and_llm.py` script when querying the *easy* dataset.
The Jupyter notebook goes through the process of visualizing the dataset and shows the design decisions we took when trying to balance the dataset.

## *Hard* Subset Creation
Similarly, the *hard* subset is made using the notebook called `create_hard_dataset_to_query.ipynb`, which emits the *hard* subset into file `hard_kernels_to_inference_unbalanced_with_compile_commands.csv`.

## Dataset Amalgamation
For convenience of future use, we provide the script `visualize_easy_and_hard_datasets_for_paper.ipynb` which does some extra visualizations of the datasets and emits the `gpuFLOPBench.csv` file which contains the whole benchmark in one CSV file.



# LLM Querying (`/gpu-flopbench/llm-querying`)

Once the datasets are created, we can begin making LLM queries. 
We designed the `run_llm_queries.py` script to accept multiple input arguments to control the type of experiment you want to run.
It is dependent on a few environment variables, depending on your LLM service provider. 
We tested only with [OpenRouter](https://openrouter.ai/) and [Microsoft Azure AI](https://ai.azure.com/) to perform our runs.
Before continuing, we want to note that these queries can easily cost hundreds of dollars depending on the model used.
Feel free to interrupt the scripts as they run to check your balances and cost of queries; we provide an estimate based on current 2025 prices which you can manually update in the `io_cost.py` file.

## OpenRouter Querying
For OpenRouter querying to work, please set the following environment variable:
```
export OPENAI_API_KEY=sk-or-v1-b5e0bed80...
```
This should be the API key you get from the OpenRouter UI.

Then you can run the following commands:
```
python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --numTrials 3 --verbose 2>&1 | tee -a ./gpt-5-mini-easy-simplePrompt.log

python3 ./run_llm_queries.py --skipConfirm --modelName openai/gpt-5-mini --numTrials 3 --verbose --hardDataset 2>&1 | tee -a ./gpt-5-mini-hard-simplePrompt.log
```
The first command will do runs with the *easy* dataset, using the `gpt-5-mini` model, while the second command will use the *hard* data subset.
The queries and outputs will be logged to the specified `*.log` file, while the full Langgraph conversations will be stored in the `./checkpoints` directory.
This process typically takes 10+ hours for one script, so please leave it running overnight or with a babysitter.
It is inherently serial, making one query at a time, so you could run both scripts simultaneously to cut down on wait times.

There is a restart mechanism in place (in case of an unexpected crash).
We suggest re-running the script after doing a first pass collection, this is due to the fact that sometimes the OpenRouter requests time out or completely fail and thus need to be re-run.

NOTE: we set a limit on the maximum query time of 2 minutes. 
If a model doesn't return, we consider it a failure.
2 minutes is quite reasonable given that a user probably wouldn't wait that long for a response anyways.

## Microsoft Azure Querying
For Azure querying to work we need to supply a particular environment variable:
```
AZURE_OPENAI_API_KEY=...
```

We can then similarly run the *easy* and *hard* data collection scripts for `o3-mini` as follows:
```
python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName o3-mini --numTrials 3 --top_p 1.0 --temp 1.0 --verbose 2>&1 | tee -a ./o3-mini-simplePrompt-easyDataset.log

python3 ./run_llm_queries.py --useAzure --api_version 2025-01-01-preview --provider_url  https://galor-m8yvytc2-swedencentral.cognitiveservices.azure.com --skipConfirm --modelName o3-mini --numTrials 3 --top_p   1.0 --temp 1.0 --verbose --hardDataset 2>&1 | tee -a ./o3-mini-simplePrompt-hardDataset.log
```

Be sure to replace the `--provider_url` with the corresponding URL from your Azure AI account.
You will also need to provide the corresponding `--api_version` from the Azure link.
Although the models we test above have a hard-coded `--top_p` and `--temp` arguments, these are just provided as-is so the Azure API would allow us to connect and run.
Any other values would return invalid request errors.


## Results Visualization / Tabulation

The final results can be visualized using the `visualizeSQLResults.ipynb` notebook.
This will calculate the Matthews Correlation Coefficient (MCC) and Mean Absolute Log Error (MALE) errors of the predictions.
It creates the plots shown in our paper, along with varying additional visualizations that aid in data analysis.

<br/><br/> <br/><br/>



# Solo (no Docker) Building & CUDA Profiling Instructions

Below is a list of instructions for reproducing what is done in the above Docker container, but instead on your own system.
This is primarily for those that don't want any overhead in GPU profiling through a Docker container (i.e: more accurate profiling results).
This path is laden with more unexpected complications and potentially more debugging effort, so continue at your own risk.
A lot of the CUDA codes we built had their compilation instructions tailored to our particular system, so you may end up having to do more work to get all the codes built and running if you decide to change compiler, compiler versions, or CUDA versions.
In future work we would like to make this process of building the codes agnostic to the system, but for now this is what we have working.

## Building

Start by simply cloning our repo.
```
git clone git@github.com:Scientific-Computing-Lab/gpuFLOPBench.git ./gpu-flopbench
cd ./gpu-flopbench
```

Execute the following command to get the Makefile generated and to start the build process.
This will automatically `make` all the programs, **you'll NEED to edit the `runBuild.sh` script to properly set any compilers/options for the codes to build**.
By default, we have everything building with `clang++` and `clang`, this should mostly work out-of-the-box but some include paths may need to be set/overriden. (SEE BELOW)

```
source ./runBuild.sh
```
NOTE: If you're running this from a Docker container generated from our Dockerfile, it should work out-of-the-box.

We originally had the CUDA codes building with `nvcc`, but to be able to also build SYCL and OMP codes, we switched to just LLVM. You may still be able to build the codes with `nvcc`, but it may take some modifications to the build pipeline.
We have future plans to sample SYCL and OMP codes, but for now, this work focuses on CUDA codes.


## Common Build Issues

The biggest build issue is that `clang` isn't assigning the proper order of includes when building a CUDA or OMP program. To get around this we include some extra flags in the `runBuild.sh` script that allow you to overwrite the include directories that `clang` tries to automgically put in. An example with the `LASSEN` build flags is included to guide what directories are required for other machines. 

Here's a list of other common build issues that might help if you're encountering errors in the build process. Most of these should have been taken care of already, but we may have missed a few.

- .c files intended to be interpreted as C++ or CUDA
- .c/.cpp files included as headers
- sources files that needed to be added (because our script didn't catch them)
- missing includes
- missing preprocessor defines
- source files that need to be unzipped
- missing libs to link
- putting some search/include dirs before others when compiling (duplicate filenames can cause header include mixups)

We note that our entire build process is captured in one `CMakeLists.txt` file.
This was done purposely to be able to build all the codes in a batch manner, as having to manually go in and modify individual HeCBench Makefiles was tiresome.

Essentially, our `CMakeLists.txt` file treats each `src/*-cuda` and `src/*-omp` directory as a single CMake/Makefile target, with the corresponding output executable having the same name as its `src` directory.
We automatically include many of the sub-directories for header files. 
The reason why our `CMakeLists.txt` file is so long is because there were many codes that we had to manually modify their build process to get them to build correctly.
This took a while to do, but ultimately makes the build process much easier and manageable.

## Python Environment Setup

We used Python3 (v3.11.11) for executing our Python scripts.
The `requirements.txt` file contains all the necessary packages and their versions that should be installed prior to using any of our Python scripts.
It is strongly advised to set up a new Conda environment to not mess up the base Python installation on your system.
Take a look at our accompanying `Dockerfile` for all the steps to set up your environment.

```
conda create --name "gpu-flopbench" python=3.11.11
conda activate gpu-flopbench
pip install -r ./requirements.txt
```
NOTE: This is already done for you if you're using the supplied `Dockerfile`.

## Gathering Profiling Data

Once all the codes are built, we can start the data collection process. We have our own script called `gatherData.py` which can be invoked to gather the profiling data of each of the built programs.

```
LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH DATAPATH=$PWD/src/prna-cuda/data_tables python3 ./gatherData.py --outfile=profiling-data.csv 2>&1 | tee runlog.txt
```
NOTE: This command should work out-of-the-box if you built a container using our Dockerfile.

This will automatically invoke each of the built executables, using `ncu` (NVIDIA Nsight Compute) to profile each of the kernels in the executable. Some of the codes require files to be downloaded proir, this script takes care of the downloading process and makes sure that all the requested files are in place.
The `DATAPATH` environment variable is only needed by `frna-cuda` and `prna-cuda`, so if you're not running those, you can drop it.
The `LD_LIBRARY_PATH` environment variable is for all the OMP codes, this is the path to the `libomptarget.so` library. On some machines CMake isn't adding the path, so we just manually add it. We should probably `rpath` this in in the future, but for now this is fine.

The internal workflow at a high level looks like the following:
1. Download rodinia dataset (skip if requested with `--skipRodiniaDownload`).
2. Gather runnable targets by scanning for executables in the `./build` directory.
3. From the gathered targets, find the ones that need extra files to be downloaded, and download/unzip them.
4. Extract the first-found execution arguments of each executable from their respective Makefiles.
5. Correct the malformed execution arguments for some targets.
6. Search for (and confirm) the existence of required input files for some programs. Unzip and extract any files that are zipped and came with HeCBench.
7. Use `cuobjdump` and `cu++filt` to extract kernel names from each executable. These are used when invoking `ncu` to profile a particular kernel.
8. Run each of the executables and gather their profiling performance data with `ncu`.
9. Write gathered data to output `profiling-data.csv` file.


The `gatherData.py` script will emit a CSV file called `profiling-data.csv` containing all the benchmarking data. After each kernel is run, the data is written out to the last line of the CSV file. We encourage writing the results of the execution to a log file for later error/execution analysis. 

‼️‼️This process of profiling all the codes can take a while (roughly 10 hours), we suggest leaving the profiling running while someone babysits in case of an unexpected error. ‼️‼️


# Limitations

1. We're not profiling all the possible kernel invocations, only the first two invocations of each kernel. There are some codes like `bitpermute-cuda` which make multiple increasing calls to its kernels, we only profile the first two.

2. This system was heavily designed around NVIDIA/CUDA GPU hardware, we have eventual plans to expand to AMD GPUs


### Future Features (TODO)
These are features we would like to have, but they're not a priority at the moment because what we have so far is giving us a good amount of data.
These also include directions we would like to move into though.

- for targets with multiple `run` makefile invocations, store all to invocations run (instead of just the first)
- support for weird precisions? -- what CUDA counters do we need?
- can we do the `ncu` regex with all the kernels -- so we just need to do one run instead of a run for each kernel
- add `nvprof` support for systems that `ncu` can't gather on (e.g: LLNL Lassen)
- perform a trial run with `nvprof`, gather all kernel launches with different launch bounds, use `ncu` `-skip` flag to target profiling each launch
  - this will gather more data as some kernels change grid-size and block-size between calls
  - this may be slower to gather data though
- figure out why some programs are having memory allocation issues (can we give different input?)
- Switch performance counter Python reading interface to use [`ncu_report`](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)
- Use `tree-sitter` to scrape CUDA kernels instead of using whole source codes
- For some weird reason, none of the APIs we use return the *thoughts* of the reasoning model, we need to include this to properly diagnose why the models mispredict
- Clean up the CMakeLists.txt logic for including files
- Only a few codes don't have their `main(...)` included in the source scrape due to us not including source files from other source directories.
- Our version of LangGraph and LangChain libraries became quickly outdated in the process of this research; we opted to keep our original versions as everything was working, therefore we need to update our scripts to better match the 1.0 release of LangGraph/LangChain.
- Extend the work to predict better on hidden FLOP counts.
- Extend the work to predict OpenMP FLOP counts (this is partly why we built with `clang`, to have one compiler do it all).