# Developmental Negation Processing in Transformer Language Models
Code for reproducing the results of our ACL 2022 paper. Scripts are under the <code>developmental_negation</code> directory.

## Reproducing Results
We have provided shell scripts for reproducing results for each experiment. In addition, under the <code>parse_nli</code> directory, we have provided all the datasets used in these experiments. All code is based on the transformers library and can be run with any supported model in huggingface. There are a few things to keep in mind about reproducing our results:

<ol>
  <li>There is a bug in diaparser which unfortunately needs to be fixed manually before the parser can be run. See this issue for details: https://github.com/Unipisa/diaparser/issues/9</li>
  <li>When saving model checkpoints, simpletransformers will name each folder as checkpoint-X-epoch-Y, where X is the training step. To not interfere with loading saved models in subsequent scripts, this should be changed to epoch-Y (this can be done recursively from the <code>checkpoint-results</code> directory).</li>
</ol>

Datasets are under <code>developmental-negation/parse_nli/nli_datasets</code>. Please note that, in addition to citing our paper, you should also cite the original authors of the SNLI and MNLI datasets, if you use this data. Each shell script under <code>developmental_negation</code> has been set up to point to the correct dataset file(s) to use for each experiment.

## About the Datasets
Each example has the same label and premise/hypothesis as it appears in the original SNLI or MNLI dataset. Our parser adds a new vector of labels that correspond to what types of developmental negation occur in the example (if any). For example, if <i>rejection</i> occurs in an example, then that example will have <code>rejection:1</code> in the JSON string. Note that our train set contains no explicit negation markers, so every one of these labels is set to 0.


## Required Packages
1. transformers
2. diaparser
3. simpletransformers
4. pytorch
5. nltk (including wordnet)

Each can be installed using pip.

## Credits
If you use this work please cite us:

>@inproceedings{laverghetta2022developmental,
  title={Developmental Negation Processing in Transformer Language Models},
  author={Laverghetta Jr, Antonio and Licato, John},
  booktitle={Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={545--551},
  year={2022}
}
