# Repository for `Assessing LLM Reliability in Relation Extraction: An Analysis on Text Processing and Prompting`

## Summary
All code required to perform the analysis, obtain the results and generate the plots is provided in this repository. 
The datasets are already present under this folder or downloaded when required, and raw results from the LLM experiments are present in `results_llm_testing` for `SentRE`, and `results_paper_triples` for DocRE, so only plotting is necessary.

## Usage
The code is divided into different notebooks:
1. `webnlg_testing.ipynb`: Run the webNLG test dataset on a given LLM. LLMs are defined in `LLMUtils.py`.
2. `evaluate_webnlg_results.ipynb`: Evaluate saved WebNLG results, saving ROUGE scores on a `.csv` file.
3. `long_form_texts_testing.ipynb`: Save triple generation statistic for either biomedical papers (read from the provided `papers.json` file) or the DocRED subset.
4. `plot_results.ipynb`: Plot results for both the WebNLG and the long-form texts evaluation.

## Requirements
A `requirements.txt` file is provided. Of note, `torch` must be available with CUDA support.
