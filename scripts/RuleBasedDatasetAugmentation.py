"""
    For labeling and prohibition, generate augmented data using Wordnet
"""
import pandas as pd
import stanza
from nltk.corpus import wordnet as wn
from nltk.tokenize import sent_tokenize, word_tokenize
from copy import deepcopy
from random import randint

en_parser = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')


def generate_questions(premise, hypothesis, label):
    premise_parse_tree = en_parser(premise).to_dict()
    hypothesis_parse_tree = en_parser(hypothesis).to_dict()
    new_rows = []

    # get the head
    premise_head = [premise_parse_tree[0][i] for i in range(len(premise_parse_tree[0])) if premise_parse_tree[0][i]['head'] == 0][0]
    hypothesis_head = [hypothesis_parse_tree[0][i] for i in range(len(hypothesis_parse_tree[0])) if hypothesis_parse_tree[0][i]['head'] == 0][0]

    if premise_head['text'] in word_tokenize(hypothesis):
        if premise_head['upos'] == 'NOUN': # and hypothesis_head['upos'] == 'NOUN':
            synset = wn.synsets(premise_head['text'])
            if len(synset) == 0:
                return new_rows
            synset = [s for s in synset if 'n' in s._name]
            lemmas = [s.lemma_names() for s in synset]
            # for simplicity, only consider lemmas that are a single word
            lemmas = list(set([val for sublist in lemmas for val in sublist if len(word_tokenize(val.replace("_", " "))) == 1]))

            for l in lemmas:
                if l == premise_head['text']:
                    continue
                row = {'gold_label':label,'posession':0,'rejection':0,'existence':0,'labeling':0,'prohibition':0,'inability':0,'epistemic':0}
                row['sentence1'] = premise.replace(premise_head['text'], l)
                row['sentence2'] = hypothesis.replace(premise_head['text'], l)
                new_rows.append(row)

        elif premise_head['upos'] == 'VERB': #and hypothesis_head['upos'] == 'VERB':
            if premise_head['lemma'] not in ['is', 'do']: # and hypothesis_head['lemma'] not in ['is', 'do']:
                synset = wn.synsets(premise_head['text'])
                if len(synset) == 0:
                    return new_rows
                synset = [s for s in synset if 'v' in s._name]
                lemmas = [s.lemma_names() for s in synset]
                lemmas = list(set([val for sublist in lemmas for val in sublist if len(word_tokenize(val.replace("_", " "))) == 1]))

                for l in lemmas:
                    if l == premise_head['text']:
                        continue
                    row = {'gold_label':label,'posession':0,'rejection':0,'existence':0,'labeling':0,'prohibition':0,'inability':0,'epistemic':0}
                    row['sentence1'] = premise.replace(premise_head['text'], l)
                    row['sentence2'] = hypothesis.replace(premise_head['text'], l)
                    new_rows.append(row)
    
    if hypothesis_head['text'] in word_tokenize(premise):
        if hypothesis_head['upos'] == 'NOUN': # and hypothesis_head['upos'] == 'NOUN':
                synset = wn.synsets(hypothesis_head['text'])
                if len(synset) == 0:
                    return new_rows
                synset = [s for s in synset if 'n' in s._name]
                lemmas = [s.lemma_names() for s in synset]
                # for simplicity, only consider lemmas that are a single word
                lemmas = list(set([val for sublist in lemmas for val in sublist if len(word_tokenize(val.replace("_", " "))) == 1]))

                for l in lemmas:
                    if l == hypothesis_head['text']:
                        continue
                    row = {'gold_label':label,'posession':0,'rejection':0,'existence':0,'labeling':0,'prohibition':0,'inability':0,'epistemic':0}
                    row['sentence1'] = premise.replace(hypothesis_head['text'], l)
                    row['sentence2'] = hypothesis.replace(hypothesis_head['text'], l)
                    new_rows.append(row)

        elif hypothesis_head['upos'] == 'VERB': #and hypothesis_head['upos'] == 'VERB':
            if hypothesis_head['lemma'] not in ['is', 'do']: # and hypothesis_head['lemma'] not in ['is', 'do']:
                synset = wn.synsets(hypothesis_head['text'])
                if len(synset) == 0:
                    return new_rows
                synset = [s for s in synset if 'v' in s._name]
                lemmas = [s.lemma_names() for s in synset]
                lemmas = list(set([val for sublist in lemmas for val in sublist if len(word_tokenize(val.replace("_", " "))) == 1]))

                for l in lemmas:
                    if l == hypothesis_head['text']:
                        continue
                    row = {'gold_label':label,'posession':0,'rejection':0,'existence':0,'labeling':0,'prohibition':0,'inability':0,'epistemic':0}
                    row['sentence1'] = premise.replace(hypothesis_head['text'], l)
                    row['sentence2'] = hypothesis.replace(hypothesis_head['text'], l)
                    new_rows.append(row)
    return new_rows        

if __name__ == "__main__":
    categories = ["labeling","prohibition"]
    negatives = pd.read_json("/home/a/alaverghett/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/nli_negation_test_v5.json",lines=True,orient="records")

    # augmentation of existing questions
    for c in categories:
        subset = negatives[negatives[c] == 1]
        new_subset = pd.DataFrame()
        for index,row in subset.iterrows():
            # sentences should be a df that can be appended directly (no loops)
            sentences = generate_questions(row["sentence1"],row["sentence2"],row['gold_label'])
            for s in range(len(sentences)):
                sentences[s][c] = 1
            if len(sentences) != 0:
                new_subset = new_subset.append(sentences)
        
        negatives = negatives.append(new_subset)
    
    
    # this will write all generated questions
    # to get the innoculation sets we are using, slice out the first 3000 from each category
    negatives.to_json("/home/a/alaverghett/from_source/negation-psycholinguistics/scripts/developmental-negation/parse_nli/nli_datasets/nli_negation_test_v5_wordnet_augmentation.json",lines=True,orient="records")