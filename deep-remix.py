import csv
import cv2
import glob
import os
import sys
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from moviepy.editor import VideoFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from os import listdir

CLIP_LEN = 3

VIDEO_FILENAME = '../source_videos/despacito.mp4'
CLIPS_DIRECTORY = '../source_clips/'
FRAMES_DIRECTORY = '../source_frames/'

CHECKPOINT_PATH="./model2.ckpt-2000000"
VOCAB_FILE="./word_counts.txt"

# flags.DEFINE_string('matching', 'entity', 'Whether to match clips based on entity, action, or both.')

def remix_video(video_filename, clips_directory, frames_directory):
	# Cut video into 3 second clips and place each clip in clips_directory
	files = glob.glob(clips_directory + '*')
	for f in files:
	    os.remove(f)
	video = VideoFileClip(video_filename)
	for start_time in range(0, int(video.duration), CLIP_LEN):
		ffmpeg_extract_subclip(video_filename, start_time, start_time + CLIP_LEN, 
			targetname=clips_directory + str(start_time) + '.mp4')
	print('Finished extracting clips!')

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
			print("Something went wrong with clip" + (clip_filename))
	print('Finished extracting frames!')

	# Generate a caption for each frame
	filenames = [(frames_directory + filename) for filename in listdir(frames_directory)]
	print('Running model.')
	captions = caption_all(filenames)
	print('Model finished.')
	for (filename, caption) in zip(filenames, captions):
		print(filename)
		print(caption)
		print()

def caption_all(filenames):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
    model = inference_wrapper.InferenceWrapper()
    restore_fn = model.build_graph_from_config(configuration.ModelConfig(),
                                               CHECKPOINT_PATH)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(VOCAB_FILE)

  with tf.Session(graph=g) as sess:
    # Load the model from checkpoint.
    restore_fn(sess)

    # Prepare the caption generator. Here we are implicitly using the default
    # beam search parameters. See caption_generator.py for a description of the
    # available beam search parameters.
    generator = caption_generator.CaptionGenerator(model, vocab)

    captions = []

    for filename in filenames:
      with tf.gfile.GFile(filename, "rb") as f:
        image = f.read()
      top_captions = generator.beam_search(sess, image)

      sentence = [vocab.id_to_word(w) for w in top_captions[0].sentence[1:-1]]
      sentence = " ".join(sentence)
      captions.append(sentence)
    return captions

if __name__ == "__main__":
	remix_video(VIDEO_FILENAME, CLIPS_DIRECTORY, FRAMES_DIRECTORY)












