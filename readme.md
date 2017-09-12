Label the images using labelImg (from GIT)

Save under test and train folder inside image dir

Creating TF records:
	create data dir where the output of the next step will be stored 
	run : python xml_to_csv.py 

	from inside models dir run : python setup.py install

	replace label in line 31 in generate_tfrecord.py
	run : python generate_tfrecord.py --csv_input=data/train_labels.csv  --output_path=data/train.record
	run : python generate_tfrecord.py --csv_input=data/test_labels.csv  --output_path=data/test.record

Downloading model and making changes :
	sample config and info links :
		 https://github.com/tensorflow/models/tree/master/object_detection/samples/configs
		 https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md
	get required configuration file from here : https://github.com/tensorflow/models/tree/master/object_detection/samples/configs

	changes in the config file:
		num_classes -> 1
		fine_tune_checkpoint -> add path to the actual checkpoint file
		PATH_TO_BE_CONFIGURED -> data
		input_path -> data/train.record
		input_path -> data/test.record

	get the model from here : https://github.com/tensorflow/models/blob/master/object_detection/g3doc/detection_model_zoo.md

	extract the model

	make new file inside training dir called object-detection.pbtxt 
	add the following to above file :
		item{
			id : 1
			name : 'box'
		}

	Put .config file in training dir 

	copy the following to models/object_detection:
		data dir
		images dir
		extracted model dir
		training

Starting the training:

	*Note* In case of error run this from models dir: 
	set PYTHONPATH=C:\Users\Ishan\Desktop\TensorFlow\models;C:\Users\Ishan\Desktop\TensorFlow\models\slim

	change dir to models/object_detection
	Script for training: python train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_mobilenet_v1_pets.config

Firing up tensorboard:
	tensorboad --logdir=training/

Prediction:
	cd to models/object_detection

	Exporting the inference graph:
	python export_inference_graph.py \ --input_type image_tensor \ --pipeline_config_path training/ssd_mobilenet_v1_pets.config \ --trained_checkpoint_prefix training/model.ckpt-1873 \ --output_directory new_graph

open object_detection_tutorial.ipynb
MODEL_NAME -> new_graph
delete MODE_FILE
remove download base
PATH_TO_LABELS -> (  data -> training ) (object-detection.pbtxt)
remove download model para
In TEST_IMAGE_PATHS -> change the range
add images to the test_images dir

