# DeepRemix
An automated remix film generator.

# Get started
Download the following files:
- Index: https://drive.google.com/file/d/1ZqSAN3MkNHe6w2lBtRPS-KpmFBDrA_ui/view?usp=sharing
- Model: https://drive.google.com/file/d/17KIwjWC8BoozeYsClhU20MHM44Le6NT3/view?usp=sharing
- Dataset: http://data.csail.mit.edu/soundnet/actions3/split1/Moments_in_Time_256x256_30fps.zip

To generate a caption for a single image, run 
~~~~
./caption_image.sh {path}/{to}/{your}/{image}
~~~~

To extract frames from all your clips and generate captions for each frame, run
~~~~
python parse_mit_dataset.py
~~~~

# Sample image captions
![alt](https://github.com/reymbarcelo/deep-remix/blob/master/sample-gifs/1.gif)

a man is in the air on a snowboard .

![alt](https://github.com/reymbarcelo/deep-remix/blob/master/sample-gifs/2.gif)

a group of people standing in a kitchen .

![alt](https://github.com/reymbarcelo/deep-remix/blob/master/sample-gifs/3.gif)

a group of people sitting around a table with laptops .
