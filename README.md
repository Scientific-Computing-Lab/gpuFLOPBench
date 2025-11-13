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

This work was presented at the [HPDC 2025](https://hpdc.sci.utah.edu/2025/) [AI4Sys Workshop](https://ai4sys.github.io/).
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

[![Build ALL CUDA/OMP Codes](https://github.com/gregbolet/gpu-flopbench/actions/workflows/buildAllCodesGithubAction.yml/badge.svg)](https://github.com/gregbolet/gpu-flopbench/actions/workflows/buildAllCodesGithubAction.yml)

## Docker Setup Instructions

For ease-of-reproducibility, we supply a `Dockerfile` with the necessary steps to recreate our environment and dataset using your own GPU hardware.
The following is a list of steps to help you get set up and into the main bash shell of the container.

‼️‼️
We note that the base container image will take up about 15 GB of storage space, which then jumps to 40 GB when we build the container; once we start building codes and gathering profiling data, the disk usage will jump up to about 50 GB.
Please ensure your system has enough storage space before continuing.
‼️‼️

```
git clone git@github.com:gregbolet/gpu-flopbench.git ./gpu-flopbench
cd ./gpu-flopbench
docker build --progress=plain -t 'gpu-flopbench' .
docker run -ti --gpus all --name gpu-flopbench-container --runtime=nvidia -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e NVIDIA_VISIBLE_DEVICES=all gpu-flopbench
docker exec -it gpu-flopbench-container /bin/bash
```

Note: if you're on a **Windows Docker Desktop** host, be sure to enable the following:
```
NVIDIA Control Panel >> Desktop Tab >> Enable Developer Settings (make sure it's enabled)
then
NVIDIA Control Panel >> Select a Task... Tree Pane >> Developer (expand section) >> Manage GPU Performance Counters >> Allow access to the GPU performance counters to all users (make sure this is enabled)
then
restart Docker Desktop
```

## Docker Data Collection Instructions (CUDA program building & profiling)

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
LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH DATAPATH=$PWD/src/prna-cuda/data_tables python3 ./gatherData.py --outfile=roofline-data.csv 2>&1 | tee runlog.txt
```
^ This process will take about 5-6 hours, so please have someone around to babysit in case any unexpected issues arise.
We tested this on our own Docker container and had no issues.

# Solo (no Docker) Instructions

Below is a list of instructions for reproducing what is done in the above Docker container, but instead on your own system.
This path is laden with more unexpected complications and potentially more debugging effort, so continue at your own risk.
A lot of the CUDA codes we built had their compilation instructions tailored to our particular system, so you may end up having to do more work to get all the codes built and running if you decide to change compiler, compiler versions, or CUDA versions.
In future work we would like to make this process of building the codes agnostic to the system, but for now this is what we have working.

## Building

Start by simply cloning our repo.
```
git clone git@github.com:gregbolet/gpu-flopbench.git ./gpu-flopbench
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

```
conda create --name "gpu-flopbench" python=3.11.11
conda activate gpu-flopbench
pip install -r ./requirements.txt
```
NOTE: This is already done for you if you're using the supplied Dockerfile.

## Gathering Roofline Data

Once all the codes are built, we can start the data collection process. We have our own script called `gatherData.py` which can be invoked to gather the roofline benchmarking data of each of the built programs.

```
LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH DATAPATH=$PWD/src/prna-cuda/data_tables python3 ./gatherData.py --outfile=roofline-data.csv 2>&1 | tee runlog.txt
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
8. Run each of the executables and gather their roofline performance data with `ncu`.
9. Write gathered data to output `roofline-data.csv` file.


The `gatherData.py` script will emit a CSV file called `roofline-data.csv` containing all the benchmarking data. After each kernel is run, the data is written out to the last line of the CSV file. We encourage writing the results of the execution to a log file for later error/execution analysis. 

‼️‼️This process of profiling all the codes can take a while (roughly 6-7 hours), we suggest leaving the profiling running while someone babysits in case of an unexpected error. ‼️‼️


## Scraping the CUDA kernels

Once all the roofline benchmarking data has been collected, we can go ahead and scrape the sampled targets for their CUDA/OMP kernels. We do this with a script called `analysis/simpleScrapeKernels.py`, which will do the following:

1. Go through all the executables in the `build` dir and extract their kernel names via `cuobjdump` or `objdump`
2. Create a dictionary assiging to each kernel the `cat` contents of all the source files used by the target

Commands below:
```
cd ./analysis
python3 simpleScrapeKernels.py
```

The scraped output will be a file called `simple-scraped-kernels-with-sass.json` in JSON format. 
We particularly do this simple form of scraping because we're struggling to have a proper AST traversal script that can properly extract CUDA kernels from source. 
This is a future step we're working on. For now, this file contains all the source files from each executable that was built in the `build` directory.
Because this is an updated version, we include SASS code in the scrape, but these are not used in the final results of this paper.

## Pruning the Scraped Kernels

Once we have the scraped source code, we can run the `analysis/vizAndPruneScrapedKernels.ipynb` notebook to:

- check the collected data using some histogram plots of the tokenizer token counts
- calculate expected minimum query cost given a percentage of the dataset
- limits the selected codes to those with 8e3 or less tokens
- emit the following files:
  - `../dataset-gen/simple-scraped-kernels-CUDA-pruned-with-sass.json`
  - `../dataset-gen/simple-scraped-kernels-OMP-pruned-with-sass.json`

Given that some codes have very long input contexts, we drop these codes from inference/testing to save on inference/training costs.
The cap we set is at 8k tokens for now, based on an initial token count analysis done to check the max number of programs we could keep without the codes being too costly to query or verbose in tokenage.
We essentially get to keep 50% of all the CUDA and OMP codes whose roofline values were sampled.


## Creating the Roofline Dataset for Querying LLMs

Once we have the `simple-scraped-kernels-CUDA-pruned-with-sass.json`, `simple-scraped-kernels-OMP-pruned-with-sass.json`, and `roofline-data.csv` files generated, we can create the Roofline Dataset by invoking the `dataset-gen/createTrainingDatset.ipynb` notebook to emit two files:

- `train-dataset-balanced.csv` -- 80% of the sampled codes
- `validation-dataset-balanced.csv` -- 20% of the sampled codes

Each row of these CSV files contains one kernel and its scraped source code along with SASS code.
For now we don't use the SASS code, as this is for a future effort.

The notebook will go through the scraped codes and roofline performance data, performing the following actions:
- Dropping any codes that were not scraped due to the token cutoff
- Creating a Roofline plot of the collected data -- it's also the plot we use in the paper
- Limits the number of times a code can appear in the dataset to once (some codes had multiple kernels that got scraped, so we only query about 1 kernel from the program)
- Balances the dataframe to have an equal number of Bandwidth-bound (BB) and Compute-bound (CB) codes (most are BB)
-  Does an 80/20 train/validation split of the full dataset

<br>
<br>

We NOTE: depending on your GPU version/capabilities, you'll need to edit the following values in the `dataset-gen/roofline_utils.py` script:
- `gpuName = 'NVIDIA RTX 3080'` -- name of the GPU
- `baseClockHz = 1.440e9` -- base clock in units of Hertz (from vendor specs)
- `maxBandwidthTBPerSec = 0.7603` -- Max global memory bandwidth in units of TB-per-sec (from vendor specs)

[The following specs for the GPU tested in this work can be found HERE](https://www.techpowerup.com/gpu-specs/evga-rtx-3080-xc3-ultra.b8041)

- `SPinstPerCyclePerSM = 128` -- number of single-precision instructions per cycle per SM
- `DPinstPerCyclePerSM = 2` -- number of double-precision instructions per cycle per SM
- `intInstPerCyclePerSM = 64` -- number of integer instructions per cycle per SM
- `numSMs = 68` -- number of SMs
- `numFMAopPerInst = 2` -- number of Fused-Multiply-Add (FMA) operations per instruction

[The above numbers can be found in a CUDA programming guide arithmetic instructions table HERE](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#arithmetic-instructions)

These values get used in calculating operations and performance counter values, so be sure to set these correctly, otherwise the roofline plots we generate may be off.
These values are used to calculate the exact same *theoretical performance* values reported on the GPU vendor spec websites, which we then use to make our Roofline model plots.


## Querying the LLMs

To perform this step, the following files should be present:
- `dataset-gen/.openrouter-api-key` -- API key to LLM service provider, we used OpenRouter and Azure to query many different models (can instead use the `--apiKey` script flag)
- `train-dataset-balanced.csv` -- The 80% of the data to query with
- `validation-dataset-balanced.csv` -- The remaining 20% of the data to query with (gets added to the 80% data) 

We invoke the `runFewShotTrials.py` script which is designed to perform both few-shot and zero-shot trials.
Check the script source for a description of each of the input arguments that can be passed to the script. 
Example invocations are provided below:


Manual API key specification, along with zero-shot trials of the `o3-mini` reasoning model.
```
python3 ./runFewShotTrials.py --zeroShot --apiKey XXXX --modelName=openai/o3-mini
```

Use the `.openrouter-api-key` file for authentication. Query the model with each sample from the dataset for all 9 combinations of `--temps` and `--topps`.
```
python3 ./runFewShotTrials.py --modelName=openai/gpt-4o-mini --temps 0.3 0.4 0.5 --topps 0.2 0.3 0.4
```

We note that the script detects when a reasoning model is being used and so it doesn't explore the various combinations of `--temps` and `--topps` hyperparameters passed to the LLM. 


## Tabulating Results

Once multiple LLMs have been run through the roofline dataset, we will have multiple CSV results files like below.
```
>> ls zero-shot-inference-results*.csv
zero-shot-inference-results-gpt-4o-2024-11-20.csv
zero-shot-inference-results-gpt-4o-mini-2024-07-18.csv
zero-shot-inference-results-gpt-4o-mini.csv
zero-shot-inference-results-o1-mini-2024-09-12.csv
zero-shot-inference-results-o1.csv
zero-shot-inference-results-o3-mini-high.csv
zero-shot-inference-results-o3-mini.csv
```

We can use the `dataset-gen/tabulateAllResultCSVs.py` script to calculate the accuracy of the LLM predictions from these CSV files.
The script prints some summary stats of the accuracy of each LLM on predicting the dataset.

```
>> python3 tabulateAllResultCSVs.py --zeroShot
...
...
                          Model Name  Number of Samples  Joint Acc  CUDA Acc  OMP Acc
                        o3-mini-high                340      64.12     64.12    64.12
                                  o1                340      64.12     64.12    64.12
                             o3-mini                340      62.06     64.71    59.41
                     gpt-4.5-preview                340      59.71     60.00    59.41
                  o1-mini-2024-09-12                340      59.64     60.12    59.17
                   gpt-4o-2024-11-20                340      52.06     52.94    51.18
                         gpt-4o-mini                340      50.59     54.12    47.06
              gpt-4o-mini-2024-07-18                340      50.29     55.88    44.71
```

The script also emits a file called: `dataset-gen/allResultsMetrics-zeroShot.csv` with all the tabulated results and additional metrics for comparison.

## Helper Scripts 

1) `dataset-gen/writeScrapedKernelWithPromptToFile.py`
This script is mainly used as a sanity check to print the full text prompt that is used for querying an LLM.
The prompt will include the system message and the filled-in templated prompt with source code.
The output is written to a `.txt` file for viewing.

2) `analysis/visualizeGatheredData-withLaunchData.ipynb`
This script does some additional grid size and block size launch bounds distribution visualization based on the sampled kernels. 
It gives us an idea of the distribution of the execution params of the codes we are sampling, along with some additional histogram plots of the sampled performances and arithmetic intensities of each kernel.

## Limitations

1. We're not profiling all the possible kernel invocations, only the first two invocations of each kernel. There are some codes like `bitpermute-cuda` which make multiple increasing calls to its kernels, we only profile the first two.

2. This system was heavily designed around NVIDIA/CUDA GPU hardware, we have eventual plans to expand to AMD GPUs


### Future Features (TODO)
These are features we would like to have, but they're not a priority at the moment because what we have so far is giving us a good amount of data.

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
- Implement a smarter way fo detecting if an LLM is reasoning-based or not
- For some weird reason, none of the APIs we use return the *thoughts* of the reasoning model, we need to include this to properly diagnose why the models mispredict
- Clean up the CMakeLists.txt logic for including files
- Only a few codes don't have their `main(...)` included in the source scrape due to us not including source files from other source directories.
