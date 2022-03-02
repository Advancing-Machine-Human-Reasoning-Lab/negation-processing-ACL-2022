# i=0
# while [ $i -le 100 ]
# do
#     epoch=1
#     mkdir /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/checkpoint_results/diagnostic_results/pseudo_test/test_$i
#     while [ $epoch -le 11 ]
#     do
#         python finetune_transformers.py --model_name roberta --model_type /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/checkpoint_results/roberta-base/epoch-$epoch --epochs 10 --batch_size 32 --cuda_device 2 --finetune no --i $i
#         ((epoch++))
#     done
#     ((i++))
# done

# for finetuning experiments

# epoch=1
# # mkdir /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/checkpoint_results/diagnostic_results/roberta
# while [ $epoch -le 10 ]
# do
#     python finetune_transformers.py --model_name roberta --model_type /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/checkpoint_results/roberta-large/epoch-$epoch --epochs 10 --batch_size 32 --cuda_device 1 --finetune no
#     ((epoch++))
# done

##################################################################################################################################
# for parsing experiments
python parser.py --input /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/multinli_1.0_train.jsonl --output /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/
python parser.py --input /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/multinli_1.0_dev_matched_cleaned.jsonl --output /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/
python parser.py --input /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/snli_1.0_dev_cleaned.jsonl --output /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/
python parser.py --input /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/snli_1.0_test_cleaned.jsonl --output /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/
python parser.py --input /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/snli_1.0_train_cleaned.jsonl --output /home/antonio/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/for_parser_v2/