#https://www.youtube.com/watch?v=cSKfRcEDGUs&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=6
#https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/#0
#https://www.tensorflow.org/tutorials/image_retraining

# Install tensorflow-hub
#--------------------
# pip install tensorflow-hub

# Download the training images
#--------------------
# on cmd console
# go on path .\06-Machine Learning Recipes #6\train\
# curl -LO http://download.tensorflow.org/example_images/flower_photos.tgz
# tar xzf flower_photos.tgz

# Download retraining script
# --------------------
# on cmd console
# go on path .\06-Machine Learning Recipes #6\
# curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/image_retraining/retrain.py

# Retrain an Image Classifier
# on bash console (I use git bash)
# go on path .\06-Machine Learning Recipes #6\
'''
python retrain.py \
  --bottleneck_dir=train/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=train/models/ \
  --summaries_dir=training_summaries/"${ARCHITECTURE}" \
  --output_graph=retrained_graph.pb \
  --output_labels=retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=train/flower_photos
'''

# Download labeler script
# --------------------
# on cmd console
# go on path .\06-Machine Learning Recipes #6\
# curl -LO https://github.com/tensorflow/hub/raw/r0.1/examples/label_image/label_image.py


# Test the algorithm
# on bash console (I use git bash)
# go on path .\06-Machine Learning Recipes #6\
# cd "D:\GitHub\rcappe\ML-Recipes\06-Machine Learning Recipes #6"

'''
python label_image.py \
--graph=retrained_graph.pb \
--labels=retrained_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=test\\test-rose.jpg
'''
# or
'''
python label_image.py \
--graph=retrained_graph.pb \
--labels=retrained_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=test\\test-daisy.jpg
'''