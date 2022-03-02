"""
    Gather results on the diagnostic using the specificed transformer
    See the argparser below for details on the command line arguments
    TODO: Add type annotations to functions
    TODO: simplify command line args, make sure everything is explained clearly
"""
import pandas as pd
from simpletransformers.classification import ClassificationModel
from transformers import RobertaTokenizerFast
import torch
from os import path, listdir, mkdir, rename
import numpy as np
import argparse
import json

def DiagTrial(args: argparse.ArgumentParser):
    print(f"Starting {path.basename(args.model_type)}")
    train = pd.read_json(args.train_set,lines=True)
    dev = pd.read_json(args.dev_set,lines=True)

    label_map = {'contradiction':0,'neutral':1,'entailment':2}

    train = train[['gold_label','sentence1','sentence2']]
    dev = dev[['gold_label','sentence1','sentence2']]

    train = train.sample(frac=1).reset_index(drop=True)
    dev = dev.sample(frac=1).reset_index(drop=True)
    train.rename({'gold_label':'labels','sentence1':'text_a','sentence2':'text_b'},inplace=True,axis=1)
    dev.rename({'gold_label':'labels','sentence1':'text_a','sentence2':'text_b'},inplace=True,axis=1)
    train['labels'] = train['labels'].apply(lambda x: label_map[x])
    dev['labels'] = dev['labels'].apply(lambda x: label_map[x])

    model_args = dict(model_name=args.model_name,
        model_type=args.model_type,
        num_labels=3,
        use_cuda=args.use_cuda,
        cuda_device=args.cuda_device,
        args=(
        {
        'output_dir':f'./checkpoint_results/{path.basename(args.model_type)}/',
        'overwrite_output_dir': False,
        'fp16': True, # uses apex
        'num_train_epochs': args.epochs,
        'reprocess_input_data': True,
        "learning_rate": 1e-5,
        "train_batch_size": args.batch_size,
        "eval_batch_size": args.batch_size,
        "max_seq_length": args.seq_len, #175
        "weight_decay": 0.01,
        "do_lower_case": False,
        "evaluate_during_training":False,
        "evaluate_during_training_verbose":False,
        "evaluate_during_training_steps":15000,
        "use_early_stopping":True,
        "early_stopping_patience":5,
        "early_stopping_consider_epochs":True,
        "save_steps": args.save_steps,
        "n_gpu": args.num_gpus,
        "logging_steps":10,
        }))
    
    try:
        if args.output_dir != None:
            # override the default save structure
            # needed for experiment 3
            model_args["args"]["output_dir"] = args.output_dir
    except Exception:
        pass

    if args.babyberta:
        model_args['tokenizer_type'] = RobertaTokenizerFast.from_pretrained("phueb/BabyBERTa-1",add_prefix_space=True)

    # Create a TransformerModel
    model = ClassificationModel(**model_args)

    if args.finetune != 'no':
        model.train_model(train,eval_df=dev)
    
    if args.eval != 'no':

        # Evaluate the model
        diagnostic = pd.read_json(args.test_set,lines=True,orient="records")
        diagnostic.rename({'gold_label':'labels','sentence1':'text_a','sentence2':'text_b'},inplace=True,axis=1)
        diagnostic['labels'] = diagnostic['labels'].apply(lambda x: label_map[x])

        result, model_outputs, wrong_predictions = model.eval_model(diagnostic)
        model_outputs = pd.Series(np.argmax(model_outputs,axis=1))
        correct_preds = pd.Series(model_outputs == diagnostic['labels'])
        diagnostic['Correct'] = correct_preds

        name = args.model_name.split("/")[-1]
        print(f"MCC: {result['mcc']}")

        diagnostic.to_csv(f"{args.diag_save_dir}/challenge_nli_{args.category}_{name}.csv")
        with open(f"{args.diag_save_dir}/challenge_nli_{args.category}_{name}_metrics.json", "w+") as outf:
            json.dump(result,outf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a Negation diagnostic experiment using Transformers')
    parser.add_argument('--finetune', action='store', type=str, default="no")
    parser.add_argument('--model_type', action='store', type=str, required=True, help='The name of the class of model architectures (BERT, etc), must be supported by simpletransformers.')
    parser.add_argument('--model_name', action='store', type=str, required=True, help='The specific model to load, either huggingface id or path to local dir.')
    parser.add_argument('--epochs', action='store', type=int, default=10, help='Number of finetuning epochs, ignored if not finetuning.')
    parser.add_argument('--batch_size', action='store', type=int, default=16, help='Finetuning batch size')
    parser.add_argument('--seq_len', action='store', type=int, default=175, help='Max seq length for finetuning.')
    parser.add_argument('--cuda_device', action='store', type=int, required=True, help="Device to run experiment on.")
    parser.add_argument('--num_gpus', action='store', type=int, default=1, help='Number of gpus to use.')
    parser.add_argument('--eval', action='store', type=str, default="no")
    parser.add_argument('--use_cuda', action='store_true', default=False)
    parser.add_argument('--train_set', action='store', type=str, required=True, help="Path to training dataset")
    parser.add_argument('--dev_set', action='store', type=str, help="Path to dev set.")
    parser.add_argument('--diag_save_dir', action='store', type=str, help="The directory where diagnostic results should be saved")
    parser.add_argument('--test_set', action='store', type=str)
    parser.add_argument('--category', action='store', type=str, help='When evaluation, the category name for the diagnostic (should match the actual file name)')
    parser.add_argument('--babyberta', action='store_true', default=False, help='Passed whenever using the babyberta model to use the correct tokenizer')
    parser.add_argument('--experiment', action='store', type=str, help='Specify the experiment configuration to use. Otherwise, just trains on the provided file')
    parser.add_argument('--output_dir', action='store', type=str, default=None)
    # save steps should be set to 2000 to replicate results from second experiment
    parser.add_argument('--save_steps', type=int, action='store', default=-1, help="How many steps to save model")
    args = parser.parse_args()

    # only meant for testing, run the script without args for training
    if args.experiment == "1":
        old_epoch = args.model_name.split("/")[-1]
        for epoch in range(1,11):
            args.model_name = args.model_name.replace(old_epoch,f"epoch-{epoch}")
            args.diag_save_dir = args.diag_save_dir.replace(old_epoch,f"epoch-{epoch}")
            DiagTrial(args)
            old_epoch = args.model_name.split("/")[-1]

    # this is only meant for evaluation, to train the models run the corresponding shell script
    # TODO: this can be merged with logic for experiment 2 test, they are almost identical
    elif args.experiment == "2":
        model = args.model_name # save the root level
        test_set_dir = args.test_set
        cats= ['posession','existence','labeling','prohibition','inability','epistemic','rejection']
        other_cats = ['posession','existence','labeling','prohibition','inability','epistemic','rejection']
        for c_i in cats:
            epochs = [e for e in listdir(path.join(model,c_i)) if "epoch" in e]
            for e in epochs:
                args.model_name = path.join(model,c_i,e)
                args.diag_save_dir = path.join(model,c_i,e)
                for c_j in other_cats:
                    args.category = c_j
                    file_name = f"{c_j}_test_innoculation_v5.jsonl"
                    args.test_set = path.join(test_set_dir,file_name)
                    args.train_set = args.test_set
                    args.dev_set = args.test_set
                    DiagTrial(args)

    elif args.experiment == "3":
        # meant for training
        if args.finetune == "yes" and args.eval == "no":
            model = args.model_name # save the root level
            train_set_dir = args.train_set
            cats = ['epistemic','existence','rejection','posession','labeling','prohibition','inability']
            for c in cats:
                file_name = f"{c}_train_innoculation_v5.jsonl_undersampled"
                args.train_set = path.join(train_set_dir,file_name)
                args.dev_set = args.train_set
                mkdir(path.join(args.diag_save_dir,c))
                args.output_dir = path.join(args.diag_save_dir,c)
                DiagTrial(args)
                epoch = [f for f in listdir(path.join(args.diag_save_dir,c)) if "epoch-10" in f]
                rename(path.join(args.diag_save_dir,c,epoch[0]),path.join(args.diag_save_dir,c,'epoch-10'))
                args.model_name = path.join(args.diag_save_dir,c,'epoch-10')
        
        elif args.eval == "yes" and args.finetune == "no":
            model = args.model_name # save the root level
            test_set_dir = args.test_set
            cats = ['epistemic','existence','rejection','posession','labeling','prohibition','inability']
            for c_i in cats:
                for e in range(1,11):
                    args.model_name = path.join(model,c_i,f"epoch-{e}")
                    args.diag_save_dir = path.join(model,c_i,f"epoch-{e}")
                    for c_j in cats:
                        args.category = c_j
                        file_name = f"{c_j}_test_innoculation_v5.jsonl"
                        args.test_set = path.join(test_set_dir,file_name)
                        args.train_set = args.test_set
                        args.dev_set = args.test_set
                        DiagTrial(args)
                        


    else:
        DiagTrial(args)