# Age Estimation New

## Subject 

This repository subject is : age estimation from face images (diff based)


## Differential Age Estimation Paper Project

Ep3 Pipeline - implementing the "Differential Age Estimation" paper - final outcome of the research.

## Other Contents (Developed During Research)

Currently:
- DiffPipeline1-3 - old, don't run. only for reference
- DiffPipeline6 - age difference estimation pipeline, with multi-ref support, with ordinal classification - LATEST ONE (USE THIS FOR RADIUS-BASED DIFF ESTIMATION)
- DiffPipeline4 - age difference estimation pipeline, embedding research notebooks, with ordinal classification. MOSTLY FOR REFERENCE - DON'T RUN
- RangeClassOrdinalPipeline - age difference classification pipeline, with ordinal classification


## Methodology

In the repository the methodology is pipeline-oriented. We develop various pipeline with the assumption they might acculate differentation with time and because we want a manageable way to track them and get back to work with each. 

Thus we don't care so much for code duplication. We prefer every ML pipeline to be very clean on its own.

An ML pipeline may have different configuration - but for "small" attributes with limited impact (not affecting the entrie structure of solution). However different pipelines differ from one another by deep and key difference (e.g. regular classification vs ordinal). I.e. different pipeline differ from one another by key hyper-parameter categories, that might affect the code deeply. In some cases, it may become a redundant 
approach (code duplication) - but this is something we accept as we prefer code "purity", high-readability and simplicity over generality. The correct way to use this approach is to separate deeply different functionality, that will result in complex "if-then" code to separate pipeline that are to be checked individually.


## Project 

Notes: this is a suggestion for standard. There may be deviation in the specific pipeline.
Go to pipeline specific README for exact directions.

Some key technical guidelines:

1. Every ML pipeline has a dir of its own.
2. Every ML pipeline starts as a flat project. It had the following files:
- <pipeline name>__config.py
- <pipeline name>__main.py
- <pipeline name>__train.py
- <pipeline name>__dataset.py
- <pipeline name>__model.py
- README.md
3. Extensions for ML pipeline - diff age pipelines
- <pipeline name>__infer.py - inference runner only for diff
- <pipeline name>__infer2.py - inference runner for age e2e (also there can be options here - bypass / full). Bypass == only original AgePredict

## General Scripts
- analyze_reg_model.py - analyzes reg model (without diff fix) - 
- analyze_reg_model_on_actual_pipe - same but using Pipeline that is used in the *__infer2.py. This is used for comparisons with the *__infer2.py when configured to bypass (Since there are small deviations). 

## Running Inference

1. Update path to weights in AgePredictBasicPipeline / assign via the API as done in DiffPipeline4 
2. Update the config file to relevant config (take from the weights dir)
3. Run <pipeline name>__infer*.py.




