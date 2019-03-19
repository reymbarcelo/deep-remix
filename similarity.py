import csv
import nltk
import operator
import pickle
import random

ENTITY_DICT = 'entity_dict.pkl'
ACTION_DICT = 'action_dict.pkl'
VOCAB_FILE = "word_counts.txt"
CAPTION_MAP = 'caption_to_clip.csv'

def map_description_to_clip(caption, entity_dict, action_dict, vocab, caption_map, matching):
	# Given a clip description, return an appropriate remix clip's filename.
	if matching == "entities":
		return map_description_to_clip_entities(caption, entity_dict, vocab)
	elif matching == "actions":
		return map_description_to_clip_actions(caption, action_dict, vocab)
	# TODO: match on both
	elif matching == "bleu":
		return map_description_to_clip_bleu(caption, caption_map)
	return None

def generate_maps():
	with open(ENTITY_DICT, 'rb') as f:
			entity_dict = pickle.load(f)
	with open(ACTION_DICT, 'rb') as f:
			action_dict = pickle.load(f)
	vocab = {}
	with open(VOCAB_FILE) as f:
		for line in f:
			(word, freq) = line.split()
			vocab[word] = int(freq)
	caption_map = []
	with open(CAPTION_MAP) as caption_file:
		caption_reader = csv.reader(caption_file, delimiter=',')
		for row in caption_reader:
			caption_map.append((row[0],row[1]))
	return entity_dict, action_dict, vocab, caption_map

def map_description_to_clip_entities(caption, entity_dict, vocab):
	# Return the clip which contains the most shared entities, 
	# ranked by rarity.
	entity_set = extract_entities(caption)
	entity_set = [entity for entity in entity_set if (entity in vocab)]
	entity_set = sorted(entity_set, key=lambda entity: vocab[entity])

	# TODO: better error handling
	if len(entity_set) == 0 or entity_set[0] not in entity_dict or len(entity_dict[entity_set[0]]) == 0:
		return ('MIT_data/training/speaking/qzHOOaLGwrw_158.mp4', 'ERROR')

	best_clip = random.choice(entity_dict[entity_set[0]])
	candidate_clips = set(entity_dict[entity_set[0]])
	matched_entities = [entity_set[0]]
	for entity in entity_set[1:]:
		candidate_clips = candidate_clips & set(entity_dict[entity])
		if len(candidate_clips) > 0:
			matched_entities.append(entity)
			best_clip = random.choice(list(candidate_clips))
	return (best_clip, ' '.join(matched_entities))

def extract_entities(caption):
	# Generate the entities in each caption
	return [noun for (noun, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[:2] == 'NN']

def map_description_to_clip_actions(caption, action_dict, vocab):
	# Return the clip which contains the most shared actions, 
	# ranked by rarity.
	action_set = extract_actions(caption)
	action_set = [action for action in action_set if (action in vocab)]
	action_set = sorted(action_set, key=lambda action: vocab[action])

	# TODO: better error handling
	if len(action_set) == 0 or action_set[0] not in action_dict or len(action_dict[action_set[0]]) == 0:
		return ('MIT_data/training/speaking/qzHOOaLGwrw_158.mp4', 'ERROR')

	best_clip = random.choice(action_dict[action_set[0]])
	candidate_clips = set(action_dict[action_set[0]])
	matched_actions = [action_set[0]]
	for action in action_set[1:]:
		candidate_clips = candidate_clips & set(action_dict[action])
		if len(candidate_clips) > 0:
			matched_actions.append(action)
			best_clip = random.choice(list(candidate_clips))
	return (best_clip, ' '.join(matched_actions))

def extract_actions(caption):
	# Generate the actions in each caption
	return [verb for (verb, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[:2] == 'VB']

def map_description_to_clip_bleu(caption, caption_map):
	best_clip = ''
	best_score = -1.0
	best_caption = ''
	split_caption = nltk.word_tokenize(caption)
	for (candidate_caption, candidate_clip) in caption_map:
		split_candidate_caption = nltk.word_tokenize(candidate_caption)
		curr_score = nltk.translate.bleu_score.sentence_bleu([split_caption], split_candidate_caption)
		if curr_score > best_score:
			best_clip = candidate_clip
			best_score = curr_score
			best_caption = candidate_caption
	return best_clip, candidate_caption










