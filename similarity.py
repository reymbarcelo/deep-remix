import nltk
import operator
import random

def map_description_to_clip(caption, entity_dict, vocab, matching):
	# Given a clip description, return an appropriate remix clip's filename.
	if matching == "entities":
		return map_description_to_clip_entities(caption, entity_dict, vocab)
	# TODO: match on action (ha)
	# TODO: match on both
	# TODO: use BLEU score
	return None

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