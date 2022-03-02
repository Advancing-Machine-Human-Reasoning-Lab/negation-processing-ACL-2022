"""
	Parser for extracting negated NLI questions, based on the work of Liu and Jasbi (2021)
	https://github.com/zoeyliu18/Negative_Constructions
"""
import io, os, string, argparse
import nltk
from diaparser.parsers import Parser
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
puncts = list(string.punctuation)



def rejection(aux):
	lemmatizer = WordNetLemmatizer()
	sent = aux[1]
	root_index = aux[7].index('root')
	lemma = lemmatizer.lemmatize(sent[root_index].lower())
	if lemma == 'like' or lemma == 'want':
		if sent[root_index-1] == 'no' or sent[root_index-1] == 'not' or sent[root_index-1] == "n't":
			
			return True
	
	return False


def epistemic(aux):
	sent = aux[1]
	root_index = aux[7].index('root')
	if sent[root_index].lower() == 'know' or sent[root_index].lower() == 'think' or sent[root_index].lower() == 'remember':
		if "no" in sent:
			index = sent.index("no")
			if sent[index-1] == "do":
				if index+1 == root_index:
					
					return True
		
		elif "not" in sent:
			if "not" in sent:
				index = sent.index("not")
				if sent[index-1] == "do":
					if index+1 == root_index:
						
						return True
		else: #n't
			if "n't" in sent:
				index = sent.index("n't")
				if sent[index-1] == "do":
					if index+1 == root_index:
						
						return True
	
	return False


def prohibition(aux):
	lemmatizer = WordNetLemmatizer()
	sent = aux[1]
	root_index = aux[7].index('root')
	if "nsubj" not in aux[7] and "nsubj:pass" not in aux[7]:
		if "no" in sent:
			index = sent.index("no")
			if sent[index-1] == "do":
				if lemmatizer.lemmatize(sent[root_index]) not in ['like','want','know','think','remember','have']:
					
					return True

		elif "not" in sent:
			index = sent.index("not")
			if sent[index-1] == "do":
				if lemmatizer.lemmatize(sent[root_index]) not in ['like','want','know','think','remember','have']:
					
					return True

		else: #n't
			index = sent.index("n't")
			if sent[index-1] == "do":
				if lemmatizer.lemmatize(sent[root_index]) not in ['like','want','know','think','remember','have']:
					
					return True
	
	return False


def inability(aux):
	sentence = aux[1]
	root_index = aux[7].index('root')
	if "nsubj:pass" not in aux[7]:
		if "no" in sentence:
			index = sentence.index("no")
			if index+1 == root_index:
				if sentence[index-1] in ["can","could"]:
					
					return True

		elif "not" in sentence:
			index = sentence.index("not")
			if index+1 == root_index:
				if sentence[index-1] in ["can","could"]:
					
					return True
		
		else: #n't
			index = sentence.index("n't")
			if index+1 == root_index:
				if sentence[index-1] in ["can","could"]:
					
					return True
	
	return False



def labeling(aux):
	from nltk.stem import WordNetLemmatizer
	lemmatizer = WordNetLemmatizer()
	sentence = aux[1]
	root_index = aux[7].index('root')
	root_pos = nltk.pos_tag(sentence)[root_index][1]
	start = lemmatizer.lemmatize(sentence[0])

	# identity or characteristics of predicative nominal
	# negation comes before the root, and the root is a noun
	if start in ['That','It']:
		# in order to disambiguate from non-existence
		# only consider 'not' and 'n't'
		# if "no" in sentence:
		# 	index = sentence.index("no")
		# 	if index < root_index and root_pos in ['NN','NNP','NNS','NNPS']:
		# 		if sentence[index-1] == "is" or sentence[index-1] == "'s":
		# 			return True
		
		if "not" in sentence:
			index = sentence.index("not")
			if index < root_index and root_pos in ['NN','NNP','NNS','NNPS']:
				if sentence[index-1] == "is" or sentence[index-1] == "'s":
					
					return True

		elif "n't" in sentence:
			index = sentence.index("n't")
			if index < root_index and root_pos in ['NN','NNP','NNS','NNPS']:
				if sentence[index-1] == "is" or sentence[index-1] == "'s":
					
					return True
	
	return False



def posession(aux):
	sentence = aux[1]
	lemmas = ['have','has','had']
	for lemma in lemmas:
		# with have
		if lemma in sentence:
			index = sentence.index(lemma)
			# have should be the head verb
			if aux[7][index] == 'root':
				if sentence[index-1] in ["n't","not"] and sentence[index-2] == "do":
					
					return True
	
	return False

	# possessive pronoun aren't included
	# they will always be sentence fragments and are unlikely to appear in NLI


# non-existence
# TODO: 'there' should come before the negation
def existence(aux):
	ok_pos = ['RB','RBR','NN','NNP','NNS','NNPS','DT']
	sentence = aux[1]
	tags = nltk.pos_tag(sentence)
	if "there" in sentence or "There" in sentence:
		# get the index
		if "there" in sentence:
			there_index = sentence.index("there")
		else:
			there_index = sentence.index("There")

		try:
			if "no" in sentence:
				index = sentence.index("no")
				if tags[index+1][1] in ok_pos and there_index < index:
					
					return True
			
			elif "not" in sentence:
				index = sentence.index("not")
				if tags[index+1][1] in ok_pos and there_index < index:
					
					return True

			else: # n't
				index = sentence.index("n't")
				if tags[index+1][1] in ok_pos and there_index < index:
					
					return True
				
		except IndexError:
			# The neagtion is at the end of the sentence
			return False

	return False


def parse(df):
	print("Starting process")
	negations = [posession, existence, labeling, prohibition, inability, epistemic, rejection]
	### Loading models
	en_parser = Parser.load('en_ewt-electra')

	# add new columns to store the label for negation type
	for n in negations:
		df[n.__name__] = 0

	for index,row in df.iterrows():
		# print(index)
		sent1 = word_tokenize(row["sentence1"])
		sent2 = word_tokenize(row["sentence2"])
		# only process sentences containing negation
		# if both the premise and the hypothesis contain negation
		if ("no" in sent1 or "not" in sent1 or "n't" in sent1 or "No" in sent1 or "Not" in sent1):
			try:
				parse_tree = en_parser.predict(sent1, text='en').sentences[0]
				attributes = parse_tree.__dict__['values']
				# now, identify the specific type of negaton
				for func in negations:
					contains = int(func(attributes))
					df.at[index,func.__name__] = contains
			except Exception:
				continue

		if ("no" in sent2 or "not" in sent2 or "n't" in sent2 or "No" in sent2 or "Not" in sent2):
			try:
				parse_tree = en_parser.predict(sent2, text='en').sentences[0]
				attributes = parse_tree.__dict__['values']
				for func in negations:
					contains = int(func(attributes))
					df.at[index,func.__name__] = contains
			except Exception:
				continue
		
		else:
			# cases without negation will just have all zeros for the labels
			continue
	
	df.to_json(args.output + "/" + file + "_negation.jsonl",orient='records',lines=True)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	# input should be a single file containing all the sentences
	parser.add_argument('--input', type = str, help = 'Path to the directly containing json files to parse.')
	parser.add_argument('--output', type = str, help = 'Extracted negation questions.')

	global args
	args = parser.parse_args()

	# global file
	# file = args.input.split("/")[-1].split(".j")[0]

	path = args.input
	os.chdir(path)

	for file in os.listdir(path):
		if file.endswith('.csv') or file.endswith('.jsonl'):
			df = pd.read_json(os.path.join(path,file),lines=True,orient='records') #chunksize
			parse(df)
