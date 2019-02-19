import csv
import cv2
import sys
import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from os import listdir

MIT_CLIPS_PATH = "../../remix-film/Moments_in_Time_256x256_30fps/training/"
MIT_FRAMES_PATH = "./sample-video-frames/"
SAMPLE_FRAMES_PATH = "./sample-images/"

FRAME_NUM = 45

CHECKPOINT_PATH="./model2.ckpt-2000000"
VOCAB_FILE="./word_counts.txt"

def generate_frames():
	# Iterate over all clips in MIT dataset
	for action in listdir(MIT_CLIPS_PATH):
		action_dirname = MIT_CLIPS_PATH + action
		if isfile(action_dirname):
			continue
		# Extract 1 frame from each clip
		for video_filename in listdir(action_dirname + "/"):
			vidcap = cv2.VideoCapture(action_dirname + "/" + video_filename)
			success, image = vidcap.read()
			# Save frame to sample directory
			if success:
				frame_filename = MIT_FRAMES_PATH + action_dirname + "____" + video_filename[:-3] + "jpg"
				print(frame_filename)
				cv2.imwrite(frame_filename, image)
			else:
				print("Something went wrong with image" + (action_dirname + "/" + video_filename))

def generate_captions(path=MIT_FRAMES_PATH):
	caption_filename = path[:-1] + "_captions.csv"
	print(caption_filename)

	with open(caption_filename, 'w', newline='') as caption_file:
		captionwriter = csv.writer(caption_file, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
		# Iterate over all frames
		filenames = [(path + filename) for filename in listdir(path)]
		for filename, caption in zip(filenames, run_model(filenames)):
			print(filename)
			print(caption)
			captionwriter.writerow([caption, filename])

def run_model(filenames):
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

#############################
generate_captions()
