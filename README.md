
# **Step recognition**

This is the code for training and evaluation of the preception models built on the [PTG project](https://github.com/VIDA-NYU/ptg-server-ml) and developed by NYU.
It can process videos and predict task (skill) steps such as the ones related to emergency medical services.

>Note: this are the used skills:  Trauma Assessment (M1), Apply tourniquet (M2), Pressure Dressing (M3), X-Stat (M5), and Apply Chest seal (R18)

## **Install**

>Note: all this process is working in the [NYU Greene HPC](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene)

#### **Dependences**

[![CLIP](https://img.shields.io/badge/CLIP-blue?logo=openai)](https://github.com/openai/CLIP)
[![ultralytics](https://img.shields.io/badge/ultralytics-green?logo=ultralytics)](https://pypi.org/project/ultralytics/)

[![librosa](https://img.shields.io/badge/librosa-red?logo=librosa)](https://pypi.org/project/librosa/)
[![supervision](https://img.shields.io/badge/supervision-yellow?logo=supervision)](https://pypi.org/project/supervision/)
[![singuconda](https://img.shields.io/badge/singuconda-brown?logo=singuconda)](https://github.com/beasteers/singuconda)

[![fire](https://img.shields.io/badge/fire-grey?logo=fire)](https://pypi.org/project/fire/)
[![fvcore](https://img.shields.io/badge/fvcore-blue?logo=fvcore)](https://pypi.org/project/fvcore/)
[![hydra-core](https://img.shields.io/badge/hydracore-green?logo=hydra-core)](https://pypi.org/project/hydra-core/)
[![einops](https://img.shields.io/badge/einops-red?logo=einops)](https://pypi.org/project/einops/)
[![timm](https://img.shields.io/badge/timm-yellow?logo=timm)](https://pypi.org/project/timm/)
[![h5py](https://img.shields.io/badge/h5py-brown?logo=h5py)](https://pypi.org/project/h5py/)
[![wandb](https://img.shields.io/badge/wandb-grey?logo=wandb)](https://pypi.org/project/wandb/)

[![simplejson](https://img.shields.io/badge/simplejson-blue?logo=simplejson)](https://pypi.org/project/simplejson/)
[![tensorboard](https://img.shields.io/badge/tensorboard-green?logo=tensorboard)](https://pypi.org/project/tensorboard/)  
[![pathtrees](https://img.shields.io/badge/pathtrees-red?logo=pathtrees)](https://pypi.org/project/pathtree/)
[![gdown](https://img.shields.io/badge/gdown-yellow?logo=gdown)](https://pypi.org/project/gdown/)

#### **Repo**

  ```
  git clone --recursive git@github.com:fabiofelix/procedural_step_recog.git

  cd procedural_step_recog/
  pip install -e .

  cd auditory_slowfast/
  pip install -e .
  ```

## **Dataset**

All video annotations should be in a CSV file with the EPICK-KITCHENS [structure](https://github.com/epic-kitchens/epic-kitchens-100-annotations). You should also add the column `video_fps` to describe the FPS of each video annotated.

## **Preprocessing videos**

The preprocessing steps are the extraction of video frames and sound. Basically, you can execute the following commands:

  1.1 Extracting frames or sound
  ```
  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/frames/ SKILL frame true 

  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/sound/ SKILL sound true 
  ```

  1.2 `/path/to/the/video/mp4/files` should be structure such as

  ```
   |- skill desc
     Data
       |- video_id
         video_id.skill_labels_by_frame.txt
         video_id.mp4
       |- video_id   
         video_id.skill_labels_by_frame.txt
         video_id.mp4
       ...               
  ```

  1.3 Using [squash](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity) to compact the files.
  ```
  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/frames/ SKILL frame false 

  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/sound/ SKILL sound false  
  ```  

>Note: to execute this script, consider to use singularity with the image *ubuntu-22.04.3.sif*  or *rockylinux-9.2.sif* both available on the NYU HPC.

## **Training and making predictions**  

Check the configuration files under `config` folder.
The field `TRAIN.ENABLE` should be *True* for training and *False* for prediction.
If you are evaluating the models, the config file should point to the model used for predictions `MODEL.OMNIGRU_CHECKPOINT_URL`.

```
bash scripts/omnimix.sh /path/to/the/frames/squash/files /path/to/the/sound/squash/files config/M2.yaml
```

>Note: this code consider the squash files previously created.

## **Visualizing the results**    

The configuration file should also point to the model used for prediction.

```
python step_recog/full/visualize.py /path/to/the/video/mp4/file output.mp4 config/M3.yaml
```

## **Feature extraction**    

The configuration file should also point to the model used for prediction and to a place to save the features `OUTPUT.LOCATION`.

```
python tools/test.py --cfg config/M3.yaml
```

## **Code structure**

1. Main code: `toos/run_step_recog.py` (function *train_kfold*)
2. Training/evaluation routines: `step_recog/iterators.py` (functions *train*, *evaluate*)
3. Model classes: `step_recog/models.py`
4. Dataloader: `step_recog/datasets/milly.py` (class *Milly_multifeature_v4* and methods *_construct_loader* and *do_getitem*)
5. Image augmentation: `tools/augmentation.py` (function *get_augmentation*)
6. Basic configuration: `step_recog/config/defaults.py` (more important), `act_recog/config/defaults.py`, `auditory_slowfast/config/defaults.py`
6. Visualizer: `step_recog/full/visualize.py` implements a specific code that combines dataloading, model prediction, and a state machine. It uses the user interface with the trained models.
