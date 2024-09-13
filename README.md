
# **Step recognition**

This is the code for training and evaluation of the preception models built on the [PTG project](https://github.com/VIDA-NYU/ptg-server-ml) and developed by NYU.
It can process videos and predict task (skill) steps such as the ones related to [tactical field care](https://www.ncbi.nlm.nih.gov/books/NBK532260/).

> [!NOTE] 
> These are the used skills:  
> - (June/2024 demo) Apply tourniquet (M2), Pressure Dressing (M3), X-Stat (M5), and Apply Chest seal (R18)
> - (December/2024 demo) Nasopharyngeal Airway (NPA) (A8), Wound Packing (M4), Ventilate with Bag-Valve-Mask (BVM) (R16), Needle Chest Decompression (R19)

## **Install**

> [!NOTE] 
> All this process is working in the [NYU Greene HPC](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/greene)
>
> Consider using [singuconda](https://github.com/beasteers/singuconda) to easily use [singularity](https://docs.sylabs.io/guides/3.5/user-guide/introduction.html) in the HPC

#### **Repo**

  ```
  git clone --recursive https://github.com/VIDA-NYU/Perception-training.git  

  cd Perception-training/
  pip install -r requirements.txt
  pip install -e .

  cd auditory_slowfast/
  pip install -e .
  ```

## **Dataset**

All video annotations should be in a CSV file with the EPICK-KITCHENS [structure](https://github.com/epic-kitchens/epic-kitchens-100-annotations). You should also add the column `video_fps` to describe the FPS of each video annotated.

> [!NOTE] 
> The code is using only these columns: video_id, start_frame, stop_frame, narration, verb_class, video_fps

## **Preprocessing videos**

The preprocessing steps are the extraction of video frames and sound. Basically, you can execute the following commands:

  1.1 Extracting frames or sound
  ```
  bash scripts/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/frames/ SKILL frame true 

  bash scripts/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/sound/ SKILL sound true 
  ```

  1.2 `/path/to/the/skill desc/` should be structured such as

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

  1.3 Using [squash](https://sites.google.com/nyu.edu/nyu-hpc/hpc-systems/hpc-storage/data-management/squash-file-system-and-singularity) to compact the files in an image that can be used with singularity.
  ```
  bash scripts/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/frames/ SKILL frame false 

  bash scripts/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/sound/ SKILL sound false  
  ```  

> [!IMPORTANT] 
> to execute this script, consider using singularity with the image *ubuntu-22.04.3.sif*  or *rockylinux-9.2.sif* both available on the NYU HPC.
>
> if you are not using singularity, remember to install *ffmeg*

  1.5 If you want to run out the NYU HPC execute this script but do not forget to install *ffmeg*
  ```
  bash scripts/out_hpc/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/frames/ SKILL frame

  bash scripts/out_hpc/extract_frames.sh /path/to/the/skill desc/Data /path/to/save/the/sound/ SKILL sound
  ```  

## **Training and making predictions**  

Check the configuration files under `config` folder.

  2.1 The field `TRAIN.ENABLE` should be *True* for training and *False* for prediction.

  2.2 Change the path to the labels `DATASET.TR_ANNOTATIONS_FILE` (train), `DATASET.VL_ANNOTATIONS_FILE` (validation), `DATASET.TS_ANNOTATIONS_FILE` (test)

  2.2 If you are evaluating the models, the config file should point to the model used for predictions `MODEL.OMNIGRU_CHECKPOINT_URL`.

  2.3 You also have to configure where are your Yolo models `MODEL.YOLO_CHECKPOINT_URL` needed to extract image features.

  2.4 The following script is always running cross-validation. Inside the script, you can change `CROSS_VALIDATION="false"` to run it with a single step.
      You also have to change the config `TRAIN.USE_CROSS_VALIDATION`.

```
bash scripts/omnimix.sh /path/to/the/frames/squash/files /path/to/the/sound/squash/files config/M2.yaml
```
> [!IMPORTANT] 
> this code uses the squash files previously created.
>
> it is also expecting the use of the singuconda

  2.4 If you want to run out the NYU HPC or singularity, change the config file to point to your frame `DATASET.LOCATION` and sound `DATASET.AUDIO_LOCATION` paths. Finally, execute this python script

```
python tools/run_step_recog.py --cfg config/M2.yaml
```

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
4. Dataloader: `step_recog/datasets/milly.py` (methods *_construct_loader* and *__getitem\__*)
- class *Milly_multifeature_v4* loads video frames and returns features
- class *Milly_multifeature_v5* loads and returns (preprocessed) features 
- class *Milly_multifeature_v6* loads and returns frames
5. Image augmentation: `tools/augmentation.py` (function *get_augmentation*)
6. Basic configuration: `step_recog/config/defaults.py` (more important), `act_recog/config/defaults.py`, `auditory_slowfast/config/defaults.py`
- examples of configuration files `config/example`
7. Visualizer: `step_recog/full/visualize.py` implements a specific code that combines dataloading, model prediction, and a state machine. It uses the user interface with the trained models.
