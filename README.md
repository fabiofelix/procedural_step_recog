
# **Action-step recognition**

This is the preception code of the [PTG project](https://github.com/VIDA-NYU/ptg-server-ml) developed by NYU. It can process representations of human actions to infer the current task step, i.e. steps of a cooking recipe or a medical procedure.

>Note: all this process is working on [NYU HPC](https://sites.google.com/nyu.edu/nyu-hpc)

## **Install**

#### **Dependences**

<!-- [![NumPy](https://img.shields.io/badge/numpy-green?logo=numpy)](https://pypi.org/project/numpy/) -->
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

[![librosa](https://img.shields.io/badge/librosa-blue?logo=librosa)](https://pypi.org/project/librosa/)
[![h5py](https://img.shields.io/badge/h5py-green?logo=h5py)](https://pypi.org/project/h5py/)
[![wandb](https://img.shields.io/badge/wandb-brown?logo=wandb)](https://pypi.org/project/wandb/)
[![simplejson](https://img.shields.io/badge/simplejson-yellow?logo=simplejson)](https://pypi.org/project/simplejson/)
[![tensorboard](https://img.shields.io/badge/tensorboard-red?logo=tensorboard)](https://pypi.org/project/tensorboard/)  

[![CLIP](https://img.shields.io/badge/CLIP-blue?logo=openai)](https://github.com/openai/CLIP)
[![ultralytics](https://img.shields.io/badge/ultralytics-green?logo=ultralytics)](https://pypi.org/project/ultralytics/)
[![pathtrees](https://img.shields.io/badge/pathtrees-brown?logo=pathtrees)](https://pypi.org/project/pathtree/)
[![pathtrees](https://img.shields.io/badge/gdown-yellow?logo=gdown)](https://pypi.org/project/gdown/)    

#### **Repo**

  ```
  git clone --recursive git@github.com:fabiofelix/procedural_step_recog.git

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

  1.2 Using [squash](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity) to compact the files.
  ```
  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/frames/ SKILL frame false 

  bash scripts/extract_frames.sh /path/to/the/video/mp4/files /path/to/save/the/sound/ SKILL sound false  
  ```  

>Note: to execute this script, consider to use *ubuntu-22.04.3.sif*  or *rockylinux-9.2.sif* available on the NYU HPC.

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

## **Code structure**

1. Main code: `toos/run_step_recog.py`
2. Training/evaluation routines: `step_recog/iterators.py` (functions *train*, *evaluate*)
3. Model classes: `step_recog/models.py`
4. Dataloader: `step_recog/datasets/milly.py` (methods *_construct_loader* and *do_getitem*)
5. Image augmentation: `tools/augmentation.py` (function *get_augmentation*)
6. Basic configuration: `step_recog/config/defaults.py` (more important), `act_recog/config/defaults.py`, `auditory_slowfast/config/defaults.py`
6. Visualizer: `step_recog/full/visualize.py` implements a specific code that combines dataloading, model prediction, and a state machine. It uses the user interface with the trained models.
