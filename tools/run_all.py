
import os, sys, argparse, logging, glob, subprocess
import pickle, pandas as pd, numpy as np

logging.basicConfig(level = logging.DEBUG)

ACTION_TYPE_EXTRACT_FRAME = "frame"
ACTION_TYPE_AUGMENT_FRAME = "augment"
ACTION_TYPE_IMG_EMBEDDING = "img"
ACTION_TYPE_SOUND_EMBEDDING = "sound"
ACTION_TYPE_ACT_RECOG     = "act"
ACTION_TYPE_STEP_RECOG    = "step"
ACTION_TYPE_SPLIT_SOUND_EMBEDDING = "split"

def extract_frames(args):
  logging.info("|- Extracting image frames from video dataset")

  if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)

  dir_list = glob.glob(os.path.join(args.source_path, "*"))
  dir_list.sort()

  for subdir in dir_list:
    dir_name = os.path.basename(subdir)
    logging.info("|--- " + dir_name)

    aux_path = os.path.join(args.output_path, dir_name)

    if not os.path.isdir(aux_path):    
      os.mkdir(aux_path)

    files = glob.glob(os.path.join(subdir, "*.mp4"))

    if len(files) > 0:
      subprocess.call(["ffmpeg", 
                      "-i", files[0],
                      "-qscale:v", "2",                     
                      os.path.join(aux_path, "frame_%010d.jpg")]) 

def augment_frames(args):
  logging.info("|- Augmentation of image frames")

  dir_list = glob.glob(os.path.join(args.source_path, "*"))
  dir_list.sort()

  for subdir in dir_list:
    dir_name = os.path.basename(subdir)

    logging.info("|--- " + dir_name)

    subprocess.call(["python", 
                    "tools/augment.py",
                    subdir,                     
                    os.path.join(args.output_path, dir_name + "_aug{}"),
                    "-n", "20"])

def extract_img_embedding(args):
  logging.info("|- Extracting embeddings from objects and frames")

  if not os.path.isdir(args.output_path):
    os.makedirs(args.output_path)

  dir_list = glob.glob(os.path.join(args.source_path, "*"))
  dir_list.sort()

  for subdir in dir_list:
    dir_name = os.path.basename(subdir)
    logging.info("|--- " + dir_name)

    subprocess.call(["python", 
                    "tools/detic_bbn.py",
                    "run_one",
                    subdir,
                    args.output_path,
                    "--skill", args.skill])

def extract_sound_embedding(args):
  logging.info("|- Extracting embeddings from sound")

  subprocess.call(["python", 
                  "auditory-slow-fast/tools/run_net.py",
                  "--cfg", args.config_file])    
  
def action_recognition(args):
  logging.info("|- Action recognition and action embeddings")

  subprocess.call(["python", 
                  "tools/run_act_recog.py",
                  "--cfg", args.config_file])  
  

def step_recognition(args):
  logging.info("|- Step recognition")

  subprocess.call(["python", 
                  "tools/run_step_recog.py",
                  "--cfg", args.config_file])  

def split_sound_features(args):
  if not os.path.isfile(args.source_path):  
    raise Exception("Source {} does not exist".format(args.source_path))
  if not os.path.isfile(args.label_path):  
    raise Exception("Label {} does not exist".format(args.label_path))    

  feature = open(args.source_path, "rb")
  feature = pickle.load(feature)    

  video_label = pd.read_pickle(args.label_path)
  frame_pattern = "frame_{:010}.npy"
  frame_idx = 45
  frame_inc = 15
  previous_video_id = None
  aux_idx = 0

  for narration_id, frame in video_label.iterrows():
    video_id, _ = frame["video_id"].split("-f")

    if previous_video_id is None or previous_video_id != video_id:
      print("Video " + video_id)
      frame_idx = 45
      aux_idx = 0

    frame_idx   += frame_inc
    output_path = os.path.join(args.output_path, video_id, "shoulders")
    feature_idx = np.where(feature["narration_id"] == str(narration_id))[0]

    if feature_idx.shape[0] > 0:
      print("|-- ({}) - Frame {} - Feature {}".format(aux_idx, frame_idx, feature_idx))

      if not os.path.isdir(output_path):
        os.makedirs(output_path)       

      numpy_file = open(os.path.join(output_path, frame_pattern.format(frame_idx)), 'wb')
      np.save(numpy_file, feature["features"][feature_idx].squeeze() if feature["features"][feature_idx].shape[0] == 1 else feature["features"][feature_idx])

    previous_video_id = video_id
    aux_idx = aux_idx + 1

def parse_action(args, parser):
  param = ""

  if args.action_type in [ACTION_TYPE_EXTRACT_FRAME, ACTION_TYPE_AUGMENT_FRAME, ACTION_TYPE_IMG_EMBEDDING, ACTION_TYPE_SPLIT_SOUND_EMBEDDING] and (args.source_path is None or args.output_path is None):  
    param = '-s/--source, -o/--output'    
  if args.action_type == ACTION_TYPE_IMG_EMBEDDING and args.skill is None:
    param += '--skill' if param == "" else ', --skill'
  if args.action_type in [ACTION_TYPE_ACT_RECOG, ACTION_TYPE_SOUND_EMBEDDING, ACTION_TYPE_STEP_RECOG] and args.config_file is None:  
    param += '--cfg' if param == "" else ', --cfg'
  if args.action_type == ACTION_TYPE_SPLIT_SOUND_EMBEDDING and args.label_path is None:  
    param += '-l/--label' if param == "" else ', -l/--label'    
    
  if param != "": 
    parser.error('the following arguments are required when --action={}: {}'.format(args.action_type, param))

  if args.action_type == ACTION_TYPE_EXTRACT_FRAME:
    extract_frames(args)
  elif args.action_type == ACTION_TYPE_AUGMENT_FRAME:
    augment_frames(args)
  elif args.action_type == ACTION_TYPE_IMG_EMBEDDING:
    extract_img_embedding(args)    
  elif args.action_type == ACTION_TYPE_SOUND_EMBEDDING:
    extract_sound_embedding(args)  
  elif args.action_type == ACTION_TYPE_ACT_RECOG:
    action_recognition(args)
  elif args.action_type == ACTION_TYPE_STEP_RECOG:
    step_recognition(args)
  elif args.action_type == ACTION_TYPE_SPLIT_SOUND_EMBEDDING:
    split_sound_features(args)    

def main(*args):
  parser = argparse.ArgumentParser(description="Conversion of dataset structure")

  parser.add_argument("-a", "--action", help = "Action type", dest = "action_type", 
                      choices = [ACTION_TYPE_EXTRACT_FRAME, 
                                 ACTION_TYPE_AUGMENT_FRAME, 
                                 ACTION_TYPE_IMG_EMBEDDING, 
                                 ACTION_TYPE_SOUND_EMBEDDING, 
                                 ACTION_TYPE_ACT_RECOG, 
                                 ACTION_TYPE_STEP_RECOG, 
                                 ACTION_TYPE_SPLIT_SOUND_EMBEDDING], required=True)
  parser.add_argument("-s", "--source", help = "Source path", dest = "source_path", required = False)
  parser.add_argument("-o", "--output", help = "Output path", dest = "output_path", required = False)
  parser.add_argument("-l", "--label", help = "Label path", dest = "label_path", required = False)  
  parser.add_argument("--skill", help = "Skill/task identifier", dest = "skill", required = False, default = None)
  parser.add_argument("--cfg", help = "Path of a needed configuration file", dest = "config_file", required = False, default = None)

  parser.set_defaults(func = parse_action)
  
  args = parser.parse_args()
  args.func(args, parser)    

#This only executes when this file is executed rather than imported
if __name__ == '__main__':
  main(*sys.argv[1:])