## Developmental Negation
<code>finetune_transformers.py</code> is the main script for running all experiments from the paper, there are various command line arguments used to run each trial. Use the provided shell scripts for the precise configuration needed for each experiment. <code>eval_experiment1_1epoch_test.sh</code> is an extra experiemnt not included in our ACL submission, and does not need to be run to reproduce what was reported in the paper. Note that
experiments are stored under a <code>checkpoint_results</code> directory and, for some experiments, the model checkpoints folders will need to be renamed to remove the "checkpoint" prefix.
This is a limitation of the simpletransformers library.

<code>Analyze_childs.ipynb</code> contains scripts for generating the plots and performing some other analyses.
