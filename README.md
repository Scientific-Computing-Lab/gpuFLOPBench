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
  
## Github Actions Status

For demonstrating the build process, we include the following Github Action in the `.github/workflows/buildAllCodesGithubAction.yml` file.
This sets up a Docker container with all the necessary programs and code to build all the HecBench codes with our CMakeLists approach.
You can copy the steps of this GitHub Action yaml file in a private Docker container to recreate the executables.

[![Build ALL CUDA/OMP Codes](https://github.com/gregbolet/HeCBench-roofline/actions/workflows/buildAllCodesGithubAction.yml/badge.svg)](https://github.com/gregbolet/HeCBench-roofline/actions/workflows/buildAllCodesGithubAction.yml)

## Building

Execute the following command to get the Makefile generated and to start the build process.
This will automatically `make` all the programs, **you'll NEED to edit the `runBuild.sh` script to properly set any compilers/options for the codes to build**.
By default, we have everything building with `clang++` and `clang`, this should mostly work out-of-the-box but some include paths may need to be set/overriden. (SEE BELOW)
```
source ./runBuild.sh
```
We originally had the CUDA codes building with `nvcc` but for simplicity have switch to just LLVM. You may be able to build the codes with `nvcc`, but it may take some modifications to the build pipeline.


# Common Build Issues

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

## Gathering Roofline Data

Once all the codes are built, we can start the data collection process. We have our own script called `gatherData.py` which can be invoked to gather the roofline benchmarking data of each of the built programs.

```
LD_LIBRARY_PATH=/usr/lib/llvm-18/lib:$LD_LIBRARY_PATH DATAPATH=/home/gbolet/hecbench-roofline/src/prna-cuda/data_tables python3 ./gatherData.py --outfile=roofline-data.csv 2>&1 | tee runlog.txt
```

This will automatically invoke each of the built executables, using `ncu` (NVIDIA Nsight Compute) to profile each of the kernels in the executable. Some of the codes require files to be downloaded proir, this script takes care of the downloading process and makes sure that all the requested files are in place.
The `DATAPATH` environment variable is only needed by `frna-cuda` and `prna-cuda`, so if you're not running those, you can drop it.
The `LD_LIBRARY_PATH` environment variable is for all the OMP codes, this is the path to the `libomptarget.so` library. One some machines CMake isn't adding the path, so we just manually add it. We should probably `rpath` this in in the future, but for now this is fine.

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


The `gatherData.py` script will emit a CSV file containing all the benchmarking data. After each kernel is run, the data is written out to the last line of the CSV file. We encourage writing the results of the execution to a log file for later error/execution analysis. 

## Scraping the CUDA kernels

Once all the roofline benchmarking data has been collected, we can go ahead and scrape the sampled targets for their CUDA/OMP kernels. We do this with a script called `analysis/simpleScrapeKernels.py`, which will do the following:

1. Go through all the executables in the `build` dir and extract their kernel names via `cuobjdump` or `objdump`
2. Create a dictionary assiging to each kernel the `cat` contents of all the source files used by the target

Commands below:
```
cd ./analysis
python3 simpleScrapeKernels.py
```

The scraped output will be a file called `simple-scraped-kernels.json` in JSON format. We particularly do this simple form of scraping because we're struggling to have a proper AST traversal script that can properly extract CUDA kernels from source. This is a future step we're working on. For now, this file contains all the source files from each executable that was built in the `build` directory.

## Building the LLM Dataset

We use the `analysis/vizAndPruneScrapedKernels.ipynb` notebook to ingest the `analysis/simple-scraped-kernels.json` file and emit two new JSON files with the pruned source data: `dataset-gen/simple-scraped-kernels-CUDA-pruned.json` and `dataset-gen/simple-scraped-kernels-OMP-pruned.json`. Given that some codes have very long input contexts, we drop these codes from inference/testing to save on inference/training costs. The cap we set is at 8k tokens for now, based on a token count analysis done to check the max number of programs we could keep without the codes being too verbose in tokenage.

After these two files have been generated, we can use the `analysis/createZeroShotDataset.ipynb` notebook to take in all three files of `dataset-gen/simple-scraped-kernels-CUDA-pruned.json`, `dataset-gen/simple-scraped-kernels-OMP-pruned.json`, and `roofline-data.csv` and emit a JSONL file for zero-shot inferencing; the file is called: `zero-shot-inference-data.jsonl`. 


## Dataset Visualization (TODO)

Once we've gathered hundreds of data samples, we want to check the data to be sure it's alright. We have some visualization scripts to help with seeing the data we collected at a high level. We do this with a Jupyter Notebook called `visualizeGatheredData-withLaunchData.ipynb` and `visualizeAndPruntScrapedKernels.ipynb`. 

## Limitations

1. We're not profiling all the possible kernel invocations, only the first two invocations of each kernel. There are some codes like `bitpermute-cuda` which make multiple increasing calls to its kernels, we only profile the first two.


### Future Features (TODO)
These are features we would like to have, but they're not a priority at the moment because what we have so far is giving us a good amount of data.

- for targets with multiple `run` makefile invocations, store all to invocations run (instead of just the first)
- support for weird precisions? -- what CUDA counters do we need?
- can we do the `ncu` regex with all the kernels -- so we just need to do one run instead of a run for each kernel
- add `nvprof` support for systems that `ncu` can't gather on
- perform a trial run with `nvprof`, gather all kernel launches with different launch bounds, use `ncu` `-skip` flag to target profiling each launch
  - this will gather more data as some kernels change grid-size and block-size between calls
  - this may be slower to gather data though
- figure out why some programs are having memory allocation issues (can we give different input?)
- Switch performance counter Python reading interface to use [`ncu_report`](https://docs.nvidia.com/nsight-compute/PythonReportInterface/index.html)
- Use `tree-sitter` to scrape CUDA kernels instead of using whole source codes
