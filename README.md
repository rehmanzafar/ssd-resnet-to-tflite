# Train and Convert SSD Resnet-50 model to TFLite
Retrain the model on your own dataset and convert it into TFLite to deploy on mobile devices and Coral Dev Board.

## Pre-requisites
- Tensorflow-gpu==1.12
- Bazel==0.24.0
- cuda==9.0
- cuDNN==7.1 

## Step 1:
Download the object_detection API from [github](https://github.com/tensorflow/models) repository or clone it by using the following command `git clone https://github.com/tensorflow/models.git`. Following is the hierarchy of the models folder: 
```
models    
│
└───research
│   │    
│   └───object_detection
│   │    
│   └───slim
│   │    
│   └───...
│  
└─── ...
```
The objective of this work is to convert the pretrained SSD Resnet-50 object detection model into TFLite, therefore only `slim` and `object_detection` directories are required from the models.

## Step 2
Download a [SSD Resnet-50](http://download.tensorflow.org/models/object_detection/ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03.tar.gz) model from a collection of pretrained models [Tensorflow Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and move it to the `object_detection` folder.
- Extract it here
- Rename the folder to `ssd-resnet-50`.
## Step 3
Set environment variable using the following command: 
```
export PYTHONPATH=$PYTHONPATH:/home/models:/home/models/research:/home/models/research/slim
```
Note: I placed the `models` in home directory. 

## Step 4
Open terminal and change your directory using `cd /home/models/research`. Further, generate protoc scripts using the following command:
```
python3.6 setup.py install
protoc object_detection/protos/*.proto --python_out=.
```

After generating the protoc scripts change working directory to `object_detection`.
```
cd /home/models/research/object_detection
```

## Step 5
Verify your tensoflow setup by running a sample script `jupyter notebook object_detection_tutorial.ipynb`, if it works you are good to go.
 
## Step 6
Annotate your data and convert xml file into a csv file using the following script:
```
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

for dir in ['train', 'test']:
    path = os.path.join(os.getcwd(), ('dataset/' + dir))
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(xml_list, columns=column_name)
    df.to_csv(('dataset/' + dir + '_labels.csv'), index=None)
    print('Successfully converted xml to csv.')
```
Edit the `generate_tfrecord.py` and verify the label mapping and execute it to generate the tfrecord file as required by the tensorflow.

```
# Change the following function in "generate_tfrecord.py"
def class_text_to_int(row_label):
    if row_label == 'cat':
        return 1
    elif row_label == 'dog':
        return 2
    else:
        None
```
 
- `python3.6 generate_tfrecord.py --csv_input=dataset/train_labels.csv --image_dir=dataset/train --output_path=train.record`
- `python3.6 generate_tfrecord.py --csv_input=dataset/test_labels.csv  --image_dir=dataset/test --output_path=test.record`

## Step 7
Generate a `labelmap.pbtxt` in training folder and define all your classes.
```
item {
  id: 1
  name: 'cat'
}

item {
  id: 2
  name: 'dog'
}
```
Copy `pipeline.config` from `ssd-resnet-50` folder to `training` folder and rename it as `ssd_resnet_50_config.config` for readability. Edit the `config` file and change the following properties:
- num_classes (Set the number of classes in your dataset)
- fine_tune_checkpoint (Set the path of `model.ckpt` from `ssd-resnet-50` folder)
- label_map_path (Set the path of `labelmap.pbtxt` from `training` folder)
- input_path (Set the path of `train.record` in `train_input_reader`)
- input_path (Set the path of `test.record` in `eval_input_reader`)
- num_examples (in eval_config)
- from_detection_checkpoint: true (Add after `fine_tune_checkpoint` in case of missing in config file)
 
## Step 8
Execute the following script to start training.
```
python3.6 train.py --logtostderr --train_dir=training/ --pipeline_config_path=training/ssd_resnet_50_config.config
```
 
## Step 9
Generate the frozen inference graph (.pb file) of SSD model.
```
python3.6 export_inference_graph.py --input_type image_tensor --pipeline_config_path training/ssd_resnet_50_config.config --trained_checkpoint_prefix training/model.ckpt-XXXX --output_directory freezed_model
``` 
Export frozen graph for TFLite.
```
python3.6 export_tflite_ssd_graph.py --pipeline_config_path=training/ssd_resnet_50_config.config --trained_checkpoint_prefix=training/model.ckpt-XXXX --output_directory=tflite_model --add_postprocessing_op=true
```
 
## Step 10
Download the source code of Tensorflow from [github](https://github.com/tensorflow/tensorflow) or clone it using `git clone https://github.com/tensorflow/tensorflow.git`.
After downloading the source code, open terminal and change working directory to `cd /home/tensorflow` and execute the following command to convert the SSD model into TFLite.
```
tflite_convert  --graph_def_file=/home/models/research/object_detection/tflite_model/tflite_graph.pb --output_file=/home/models/research/object_detection/tflite_model/detect.tflite --output_format=TFLITE --input_shapes=1,640,640,3 --input_arrays=normalized_input_image_tensor --output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' --inference_type=FLOAT --mean_values=128 --std_dev_values=127 --change_concat_input_ranges=false --allow_custom_ops
```

## Step 11
Use the `.tflite` file to deploy the model on mobile devices and Coral Dev Board.

## References
https://www.tensorflow.org/lite/convert/cmdline_examples 
