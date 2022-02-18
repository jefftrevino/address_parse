# adapted by Jeff Trevino (2022-02-18) from
# https:\\github.com\swapnil-saxena\address-parser\blob\main\training_data_prep.py
# by Swapnil Saxena, 2021-11-08

# approach: train a natural language processing model
# to perform named entity recognition (NER)

from collections import namedtuple
import re

import pandas as pd
import spacy
from spacy.tokens import DocBin

def clean_address(address):
	cleansed_address1 = re.sub(r'(,)(?!\s)', ', ', address) # spaces after all commas
	cleansed_address2 = re.sub(r'(\\n)',', ', cleansed_address1) # newlines are ', '
	cleansed_address3 = re.sub(r'(?!\s)(-)(?!\s)',' - ', cleansed_address2) # spaces around dashes
	cleansed_address = re.sub(r'\.','', cleansed_address3) # remove periods
	return cleansed_address


def get_start_and_end_index(address_string, address_component_value):
	clean_component = re.sub('\.','', address_component_value)
	clean_component = re.sub(r'(?!\s)(-)(?!\s)',' - ', clean_component)
	span = re.search(f'\\b(?:{clean_component})\\b', address_string)
	assert span, "Could not find address token in address"
	return span.start(), span.end()


def annotate_address(address_row):
	address = address_row['ADDRESS']
	annotation_tuples = []
	for key, value in address_row.items():
		if pd.notnull(value):
			if key != 'ADDRESS':
				annotation_tuples.append(build_annotation_tuple(address, key, value))
	return (address, annotation_tuples)


def build_annotation_tuple(address, key, value):
	start_index, end_index = get_start_and_end_index(address, value)
	return (start_index, end_index, key)


def get_doc_tuple_series(df):
	df['ADDRESS'] = df['ADDRESS'].apply(lambda x: clean_address(x))
	doc_tuple_series = df.apply(annotate_address, axis=1)
	return doc_tuple_series


def create_doc_bin(training_data_series, nlp):
	# a spaCy DocBin is a collectino of docs
    doc_bin = DocBin()
    for text, annotations in training_data_series.to_list():
		# a spaCy doc is a container of linguistic annotations of some text
        document = create_doc_from_text_and_annotations(text, annotations)
        doc_bin.add(document)
    return doc_bin


def create_doc_from_text_and_annotations(text, annotations):
	document = nlp(text)
	annotation_tuples = []
	print(annotations)
	for start, end, label in annotations:
		span = document.char_span(start, end, label=label)
		annotation_tuples.append(span)
	document.ents = annotation_tuples
	return document


def get_docbin_from_data(path_to_dataset):
	df_test=pd.read_csv(path_to_dataset, header=0, sep=",", dtype=str)
	df_test.columns = [c.upper() for c in df_test.columns]
	training_data_series = get_doc_tuple_series(df_test)
	doc_bin = create_doc_bin(training_data_series, nlp)
	return doc_bin


nlp = spacy.blank("en")
# --------------------=== ingest and annotate train set === --------------------
doc_bin_train = get_docbin_from_data("./data/us_address_train_dataset.csv")
doc_bin_train.to_disk("./corpus/spacy-docbins/train.spacy")

# --------------------=== ingest and annotate test set === --------------------
doc_bin_test = get_docbin_from_data("./data/us_address_test_dataset.csv")
doc_bin_test.to_disk("./corpus/spacy-docbins/test.spacy")

# after generating config.cfg from base_config.cfg (see base_config.cfg),
# train the model by calling
# python -m spacy train config/config.cfg --paths.train corpus/spacy-docbins/train.spacy --paths.dev corpus/spacy-docbins/test.spacy --output output/models --training.eval_frequency 10 --training.max_steps 300
