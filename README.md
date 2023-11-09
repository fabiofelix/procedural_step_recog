
# **Action-step recognition**

Implements (and trains) a Recurrent Neural Network that can process representations of human actions to infer the current step in a procedure (i.e. step in a recipe or in a medical procedure).

`Note: Do not forget to install this repository`

## **Dataset**

1. Prepare your video dataset

    1.1 We use the same structure of the [EPICK-KITCHENS](https://epic-kitchens.github.io/). Therefore, your videos should be in the following structure.

    | main dir |          |              |
    |----------|----------|--------------|
    |          | video1-1 |              |
    |          |          | video1-1.mp4 |
    |          | video1-2 |              |
    |          |          | video1-2.mp4 |
    |          | video1-3 |              |
    |          |          | video1-3.mp4 |

    1.2 Following, the EPICK-KITCHENS [annotation](https://github.com/epic-kitchens/epic-kitchens-100-annotations) is a pickle file with the following fields

    | field               | type        | example                        |
    |---------------------|-------------|--------------------------------|
    | participant_id      | string      | P01, P02, R01, R03             |
    | video_id            | string      | P01_01, P01_02, R01_100        |
    | narration_id        | string      | P01_01_1, P01_01_2, P01_01_100 |
    | narration_timestamp | timestamp   | 00:00:01.089                   |
    | start_timestamp     | timestamp   | 00:00:01.089                   |
    | stop_timestamp      | timestamp   | 00:00:01.089                   |
    | start_frame         | int         | 19172                          |
    | stop_frame          | int         | 19633                          |
    | narration           | string      | pour out boiled water          |
    | verb                | string      | pour-out                       |
    | verb_class          | int         | 9                              |
    | noun                | string      | water                          |
    | noun_class          | int         | 27                             |
    | all_nouns           | string list | [water]                        |
    | all_noun_classes    | int list    | [27]                           |

    1.3 You should convert it to CSV file and disconsider the fields list and timestamp

    1.4 If necessary, split your dataset into train/validation/test.

2. (optional) Finally, the video sound clips should be a HDF5 file.

    2.1 You can execute the extractor on [Auditory Slow-Fast](https://github.com/ekazakos/auditory-slow-fast).

    2.2 Generate k-length clips and give an *id* similiar to the *video_id* of the point 1.2. 
    For example, if you split your sound in 2s-length clips then the HDF5 file have inputs such as P01_01-c1, P01_01-c2, P01_01-c3, ..., P01_01-c30.

    2.2 Create also a pickle file with the same structure of 1.2. You have to provide one row for each audio clip. However, you have only to care about the id fields, leting the other fields empty.


## **Extracting features**    

1. Extract the video **frames**

    1.1 Install the following dependences

    [![FFmpeg](https://img.shields.io/badge/ffmpeg-brown?logo=ffmpeg)](https://www.ffmpeg.org/)

    1.2 Execute the script

    ```
    python tools/run_all.py -a frame -s /path/to/videos -o /path/to/output/rgb_frames
    ```
    example: `python tools/run_all.py -a frame -s /home/user/data/video -o /home/user/data/frame/rgb`

2. **Augment** video frames

    2.1. Install the following dependences 

    [![NumPy](https://img.shields.io/badge/numpy-green?logo=numpy)](https://pypi.org/project/numpy/)
    [![Opencv](https://img.shields.io/badge/opencv-brown?logo=opencv)](https://pypi.org/project/opencv-python/)
    [![pandas](https://img.shields.io/badge/pandas-blue?logo=pandas)](https://pypi.org/project/pandas/)
    [![fire](https://img.shields.io/badge/fire-yellow?logo=fire)](https://pypi.org/project/fire/)
    [![Pillow](https://img.shields.io/badge/pillow-red?logo=pillow)](https://pypi.org/project/Pillow/)
    [![fvcore](https://img.shields.io/badge/fvcore-grey?logo=fvcore)](https://pypi.org/project/fvcore/)
    [![hydra-core](https://img.shields.io/badge/hydracore-green?logo=hydra-core)](https://pypi.org/project/hydra-core/)
    [![einops](https://img.shields.io/badge/einops-brown?logo=einops)](https://pypi.org/project/einops/)
    [![torch](https://img.shields.io/badge/torch-blue?logo=torch)](https://pypi.org/project/torch/)
    [![torch-vision](https://img.shields.io/badge/torchvision-yellow?logo=torchvision)](https://pypi.org/project/torchvision/)
    [![timm](https://img.shields.io/badge/timm-red?logo=timm)](https://pypi.org/project/timm/)

    2.2 Download and install this github package

    2.3. Execute the script 
    ```
    python tools/run_all.py -a augment -s /path/to/rgb_frames -o /path/to/output/aug_rgb_frames
    ```
    example: `python tools/run_all.py -a augment -s /home/user/data/frame/rgb -o /home/user/data/frame/rgb_aug`    

3. Extract the video **object** and **frame** embeddings using [Yolo](https://github.com/ultralytics/ultralytics) and [CLIP](http://proceedings.mlr.press/v139/radford21a):


      3.1 Install the following dependences

    [![CLIP](https://img.shields.io/badge/CLIP-blue?logo=openai)](https://github.com/openai/CLIP)
    [![ultralytics](https://img.shields.io/badge/ultralytics-green?logo=ultralytics)](https://pypi.org/project/ultralytics/)
    [![pathtrees](https://img.shields.io/badge/pathtrees-brown?logo=pathtrees)](https://pypi.org/project/pathtree/)
    [![pathtrees](https://img.shields.io/badge/gdown-yellow?logo=gdown)](https://pypi.org/project/gdown/)
    
      3.2. Download the code and follow the instructions in [NYU-PTG-server](https://github.com/VIDA-NYU/ptg-server-ml) to install the package. 

      3.3. Execute the script
      ```
      python tools/run_all.py -a img -s /path/to/rgb_frames/video -o /path/to/object/features --skill skill_tag
      ```   
      example: `python tools/run_all.py -a img -s /home/user/data/frame/rgb_aug/VIDEO-1 -o /home/user/data/features/obj_frame --skill M1`

4. (optional) Extract the **sound** embeddings using Auditory Slow-Fast:

    4.1 The code inside the folder auditory-slow-fast is available on [Auditory Slow-Fast](https://github.com/ekazakos/auditory-slow-fast). 

    4.2. Install the following dependences

    [![librosa](https://img.shields.io/badge/librosa-blue?logo=librosa)](https://pypi.org/project/librosa/)
    [![h5py](https://img.shields.io/badge/h5py-green?logo=h5py)](https://pypi.org/project/h5py/)
    [![wandb](https://img.shields.io/badge/wandb-brown?logo=wandb)](https://pypi.org/project/wandb/)
    [![simplejson](https://img.shields.io/badge/simplejson-yellow?logo=simplejson)](https://pypi.org/project/simplejson/)
    [![tensorboard](https://img.shields.io/badge/tensorboard-red?logo=tensorboard)](https://pypi.org/project/tensorboard/)  

    4.3 Configure a YAML file in auditory-slow-fast/configs/ dir.

    4.4 Execute the script

    ```
    python tools/run_all.py -a sound --cfg /path/to/config/file
    ```   
    example: `python tools/run_all.py -a sound --cfg auditory-slow-fast/configs/BBN/SLOWFAST_R50.yaml`

    4.5 Aditionally, the last code generates one pickle file with all the features but our approach demands a sequence of numpy files per frame. Therefore, run the next code to split the big file in small portions.

    ```
    python tools/run_all.py -a split -s /path/to/one/pickle/feature/file -o /path/to/output/numpy/many/files -l /path/to/pickle/annotations/file 
    ```   
    example: `python tools/run_all.py -a split -s /home/user/data/features/sound/validation.pkl -o /home/user/data/features/sound/per-video/ -l /home/user/data/features/sound/annotation.pkl`

    *Note*: Check the section **Dataset** 2.2 for the annotation.pkl

5. Extract the video **action** embeddings using [Omnivore](https://arxiv.org/abs/2201.08377):

    5.1 Configure a YAML file in config/ dir.

    5.2. Execute the script

    ```
    python tools/run_all.py -a act --cfg /path/to/config/file
    ```   
    example: `python tools/run_all.py -a act --cfg config/OMNIVORE.yaml`


## **Training and making predictions**    

1. Train the video **step** recognizer:    

    1.1 Configure a file YAML file in config/ dir.

    1.2. Execute the script

    ```
    python tools/run_all.py -a step --cfg "/path/to/config" 
    ```   
    example: `python tools/run_all.py -a step --cfg config/STEPGRU.yaml`

2. Recognize the **steps**    

    2.1 Edit the YAML file in config/ dir. and run the last command again


## **Running things on the [NYU HPC](https://sites.google.com/nyu.edu/nyu-hpc)**    

HPC has restrictions about number and size of files.
Therefore, we are using [squash](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity) files to deal with the number of generated files.
The scripts in the steps 1, 2, 3, and 5 process the inputs and [squash](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity) the outputs, creating one compressed file per video. 


1. Extract the video **frames**

    1.1 Execute the script

    ```
    bash scripts/extract_frames.sh "/path/to/videos/" "/path/to/output/rgb_frames"
    ```
    example: `bash scripts/extract_frames.sh /home/user/data/video /home/user/data/frame/rgb`

    `Note: You have to run this script twice. One to extract the frames (setting extract="true" inside the script) and the second to squash the files (setting extract="false" inside the script). In the first time, run inside a singularity that has ffmpeg installed. In the second one, do not use a singularity. squash commands are not visuble in a singularity environment (we do not know why).`

2. **Augment** video frames

    2.1. Execute the script 
    ```
    bash scripts/augment.sh "/path/to/rgb_frames" "/path/to/output/aug_rgb_frames"
    ```
    example: `bash scripts/augment.sh /home/user/data/frame/rgb /home/user/data/frame/rgb_aug`    

3. Extract the video **object** and **frame** embeddings using [Yolo](https://github.com/ultralytics/ultralytics) and [CLIP](http://proceedings.mlr.press/v139/radford21a):

      3.1. Execute the script
      ```
      bash scripts/detic_bbn.sh "/path/to/augmented/rgb_frames" "/path/to/output/features" skill_tag
      ```   
      example: `bash scripts/detic_bbn.sh /home/user/data/frame/rgb_aug/ /home/user/data/features/obj_frame M1`

4. (optional) Extract the **sound** embeddings using Auditory Slow-Fast:

    4.1 Execute the script

    ```    
    ```   
    example: ` `

    4.2 Run the next code to split the big file in small portions.

    ```
    ```   
    example: ` `

5. Extract the video **action** embeddings using [Omnivore](https://arxiv.org/abs/2201.08377):

    5.1. Execute the script

    ```
    bash scripts/omnivore.sh /path/to/row/and/augmented/frame /path/of/out/features /path/to/config/file
    ```   
    example: `bash scripts/omnivore.sh /home/user/data/frame/ /home/user/data/features/omnivore/test_subset /home/user/config/OMNIVORE.yaml`

    `Note: The output path of the script is the same specified in the YAML configuration file`
    

6. Training the video **step** recognizer:    

    6.1. Execute the script

    ```
    bash scripts/omnimix.sh /path/to/action/features/ /path/to/object/and/frame/features /home/user/data/features/sound/ /path/to/config/file
    ```   
    example: `bash scripts/omnimix.sh /home/user/data/features/omnivore/ /home/user/data/features/obj_frame /home/user/data/features/sound/ /home/user/config/STEPGRU.yaml`
    

7. Recognizing the **steps**    

    7.1. Execute the previsous step but disable training in the configuration file.
    ```
    TRAIN
      ENABLE: False
    ```      
     