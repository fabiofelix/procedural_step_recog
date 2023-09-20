
# **Action-step recognition**

Implements (and trains) a Recurrent Neural Network that can process representations of human actions to infer the current step in a procedure (i.e. step in a recipe or in a medical procedure).

`Note: Run all the following commands inside the main dir of the code.`

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

3. Extract the video **object** and **frame** embeddings using [Detic](https://arxiv.org/abs/2201.02605):


      3.1 Install the following dependences

    [![CLIP](https://img.shields.io/badge/CLIP-blue?logo=openai)](https://github.com/openai/CLIP)
    [![ultralytics](https://img.shields.io/badge/ultralytics-green?logo=ultralytics)](https://pypi.org/project/ultralytics/)
    [![pathtrees](https://img.shields.io/badge/pathtrees-brown?logo=pathtrees)](https://pypi.org/project/pathtree/)
    [![pathtrees](https://img.shields.io/badge/gdown-yellow?logo=gdown)](https://pypi.org/project/gdown/)
    
      3.2. Download the code and follow the instructions in [NYU-PTG-server](https://github.com/VIDA-NYU/ptg-server-ml) to install the package. 

      3.3. Execute the script
      ```
      python tools/run_all.py -a img -s /path/to/rgb_frames -o /path/to/object/frames --skill skill_tag
      ```   
      example: `python tools/run_all.py -a img -s /home/user/data/frame/rgb_aug -o /home/user/data/features/obj_frame --skill M5`

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

    4.5 Aditionally, the last code generates one pickle file with all the features but our approach demands a sequence of numpy files per frame. Therefore, ran the next code to split the big file in small portions.

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

1. Extract the video **frames**

    1.1 Execute the script

    ```
    bash scripts/extract_frames.sh "/path/to/videos/" "/path/to/output/rgb_frames"
    ```
    example: `bash scripts/extract_frames.sh "/Users/iranroman/datasets/M2_Tourniquet/Data" "Users/iranroman/datasets/BBN_0p52/M2_Tourniquet/rgb_frames"`

2. **Augment** video frames

    2.1. Execute the script 
    ```
    bash scripts/augment.sh "/path/to/rgb_frames" "/path/to/output/aug_rgb_frames"
    ```
    example: `bash scripts/augment.sh "/vast/iranroman/BBN/M5_X-Stat/rgb_frames /vast/iranroman/BBN/M5_X-Stat/aug_rgb_frames"`    

3. Extract the video **object** and **frame** embeddings using [Detic](https://arxiv.org/abs/2201.02605):

      3.1. Execute the script
      ```
      bash scripts/detic_bbn.sh "/path/to/augmented/rgb_frames" "path/to/output/features" skill_tag
      ```   
      example: `bash scripts/detic_bbn.sh "/vast/iranroman/BBN/M2_Lab_Skills/aug_rgb_frames" "/vast/iranroman/BBN-features" M2`

4. Extract the video **action** embeddings using [Omnivore](https://arxiv.org/abs/2201.08377):

    4.1. Execute the script

    ```    
    ```   
    example: ` `

5. (optional) Extract the **sound** embeddings using Auditory Slow-Fast:

    5.1 Execute the script

    ```    
    ```   
    example: ` `

    5.2 Ran the next code to split the big file in small portions.

    ```
    ```   
    example: ` `

6. Train the video **step** recognizer:    

    6.1. Execute the script

    ```
    ```   
    example: ` `

7. Recognize the **steps**    

    7.2. Execute the script

    ```
    ```   
    example: ` `     