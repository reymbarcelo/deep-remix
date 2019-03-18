import csv
import cv2
import glob
import nltk
import operator
import os
import pickle
import sys
import tensorflow as tf

from argparse import ArgumentParser
from caption_model_utils import caption_all
from collections import defaultdict
from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from moviepy.editor import VideoFileClip, concatenate_videoclips
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip, ffmpeg_extract_audio, ffmpeg_merge_video_audio
from os import listdir


CLIP_LEN = 3

VIDEO_FILENAME = 'despacito.mp4'

VIDEO_DIRECTORY = 'remix_files/source_videos/'
AUDIO_DIRECTORY = 'remix_files/source_audio/'
CLIPS_DIRECTORY = 'remix_files/source_clips/'
FRAMES_DIRECTORY = 'remix_files/source_frames/'
REMIX_DIRECTORY = 'remix_files/remix_videos/'
CAPTION_DIRECTORY = 'remix_files/captions/'

VOCAB_FILE = "word_counts.txt"

def extract_clips(video_filename, clips_directory):
	# Cut video into 3 second clips and place each clip in clips_directory
	files = glob.glob(clips_directory + '*')
	for f in files:
	    os.remove(f)
	video = VideoFileClip(video_filename)
	for start_time in range(0, int(video.duration), CLIP_LEN):
		# TODO: silence this call's output
		ffmpeg_extract_subclip(video_filename, start_time, start_time + CLIP_LEN, 
			targetname=clips_directory + ('%04d' % start_time) + '.mp4')
	print('Finished extracting clips.')

def extract_frames(clips_directory, frames_directory):
	# Generate a frame for each clip
	files = glob.glob(frames_directory + '*')
	for f in files:
	    os.remove(f)
	for filename in listdir(clips_directory):
		clip_filename = clips_directory + filename
		vidcap = cv2.VideoCapture(clip_filename)
		success, image = vidcap.read()
		# Save frame to sample directory
		if success:
			frame_filename = frames_directory + filename[:-3] + "jpg"
			cv2.imwrite(frame_filename, image)
		else:
			print("Something went wrong with clip " + (clip_filename))
	print('Finished extracting frames.')

def generate_captions(frames_directory):
	# Generate a caption for each frame
	filenames = [(frames_directory + filename) for filename in sorted(listdir(frames_directory))]
	captions = caption_all(filenames)
	print('Finished generating captions.')

	return (filenames, captions)

def extract_entities(captions):
	# Generate the entities for each caption
	entities = []
	for caption in captions:
		caption_entities = [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(caption)) if pos[:2] == 'NN']
		entities.append(caption_entities)
	return entities

def map_description_to_clip(entity_set, entity_dict, vocab):
	# Given a clip description, return an appropriate remix clip's filename.
	# TODO: incorporate word frequency (from VOCAB_FILE)
	# TODO: use an encoder-decoder network
	clip_scores = defaultdict(int)

	# Return the clip which contains the most entities. Given sparsity of 
	# entity set, will most likely only contain 1.
	for entity in entity_set:
		# TODO: include actions
		if entity in entity_dict and entity in vocab:
			relevant_clips = entity_dict[entity]
			for clip in relevant_clips:
				clip_scores[clip] += 1
	best_clip = max(clip_scores.items(), key=operator.itemgetter(1))[0]
	return best_clip

def generate_remix_filenames(entities):
	# For all clip descriptions, find an appropriate remix clip and return its filename.
	with open('entity_dict.pkl', 'rb') as f:
			entity_dict = pickle.load(f)
	vocab = {}
	with open(VOCAB_FILE) as f:
		for line in f:
			(word, freq) = line.split()
			vocab[word] = int(freq)

	remix_filenames = []
	for entity_set in entities:
		remix_filenames.append(map_description_to_clip(entity_set, entity_dict, vocab))
	return remix_filenames

def remix_video(video_filename, clips_directory, frames_directory, verbose):
	extract_clips(VIDEO_DIRECTORY + video_filename, clips_directory)
	extract_frames(clips_directory, frames_directory)
	filenames, captions = generate_captions(frames_directory)
	entities = extract_entities(captions)
	remix_filenames = generate_remix_filenames(entities)

	if verbose:
		with open(CAPTION_DIRECTORY + video_filename.replace('mp4', 'txt'), 'w') as f:
			for filename, caption, remix_clipname in zip(filenames, captions, remix_filenames):
				f.write(filename + '\n')
				f.write(caption + '\n')
				f.write(remix_clipname + '\n')
				f.write('\n')

	# Concatenate clips together
	video_file_clips = [VideoFileClip(remix_clipname) for remix_clipname in remix_filenames]
	remix_video = concatenate_videoclips(video_file_clips)
	remix_filename = REMIX_DIRECTORY + 'remix-' + video_filename
	remix_video.write_videofile(remix_filename.replace('.mp4', '-noaudio.mp4'))
	ffmpeg_extract_audio(VIDEO_DIRECTORY + video_filename, 
		AUDIO_DIRECTORY + video_filename.replace('mp4', 'mp3'))
	# TODO: get audio to work
	ffmpeg_merge_video_audio(remix_filename.replace('.mp4', '-noaudio.mp4'), 
		AUDIO_DIRECTORY + video_filename.replace('mp4', 'mp3'), remix_filename)

###############################################################################

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--video_filename", default=VIDEO_FILENAME)
	parser.add_argument("--clips_directory", default=CLIPS_DIRECTORY)
	parser.add_argument("--frames_directory", default=FRAMES_DIRECTORY)
	parser.add_argument("--verbose", default=False)
	args = parser.parse_args()

	remix_video(args.video_filename, args.clips_directory, args.frames_directory, args.verbose)












