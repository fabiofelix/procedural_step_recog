from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb, ipdb
import json
import scipy
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sb, pandas as pd
import warnings
import glob

def build_model(cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model  = OmniGRU(cfg)
  model.to(device)

  return model, device  

def train_step(epoch, model, criterion, criterion_t, optimizer, loader, is_training, device, cfg, progress, flatten_out = False):
  if is_training:
    model.train()
    h = model.init_hidden(cfg.TRAIN.BATCH_SIZE)
  else:
    model.eval()  

  sum_class_loss = 0.
  sum_pos_loss = 0.
  sum_loss = 0.
  sum_b_acc = 0.
  sum_acc = 0.
  number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1

  for counter, (action, obj, frame, audio, label, label_t, mask, frame_idx, video_id) in enumerate(loader, 1):
    label = nn.functional.one_hot(label, number_classes)

    if not is_training:
      h = model.init_hidden(len(action))

    h = torch.zeros_like(h)
    optimizer.zero_grad()

    out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)

    mask    = mask.to(out.device)  
    label   = label.to(out.device)
    label_t = label_t.to(out.device)

    out_t  = torch.softmax(out[..., number_classes:], dim = -1)  #regression of time positions. Limits the results to [0, 1] range
    out    = out[..., :number_classes]                           #classification of steps. CrossEntropyLoss consumes logits          

    out_masked   = out * mask
    label_masked = mask * label.float()      

##   https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
##   https://discuss.pytorch.org/t/rnn-many-to-many-classification-with-cross-entropy-loss/106197
    class_loss = criterion(out_masked.transpose(1, 2), label_masked.transpose(1, 2))
    pos_loss   = criterion_t(out_t.float() * mask, mask * label_t.float())
    loss       = class_loss + pos_loss

    if is_training:
      loss.backward()
      optimizer.step()

    sum_class_loss += class_loss.item()
    sum_pos_loss   += pos_loss.item()
    sum_loss       += loss.item()

    out_masked   = torch.argmax(torch.softmax(out_masked, dim = -1), axis = -1)
    label_masked = torch.argmax(label_masked, axis = -1)

    mask         = torch.flatten(mask).cpu().numpy()  
    label_masked = torch.flatten(label_masked).cpu().numpy()  
    out_masked   = torch.flatten(out_masked).cpu().numpy()

    label_masked_aux = []
    out_masked_aux = []

    for m, l, o in zip(mask, label_masked, out_masked):
      if m > 0:
        label_masked_aux.append(l)
        out_masked_aux.append(o)

    label_masked_aux = np.array(label_masked_aux)
    out_masked_aux = np.array(out_masked_aux)
    sum_acc   += accuracy_score(label_masked_aux, out_masked_aux)      
    sum_b_acc += weighted_accuracy(label_masked_aux, out_masked_aux, number_classes)

    if is_training:
      progress.update(1)
      progress.set_postfix({"Cross entropy": sum_class_loss/counter, 
                            "MSE": sum_pos_loss/counter, 
                            "Total loss": sum_loss/counter, 
                            "acc": sum_acc/counter})

  grad_norm = np.sqrt(np.sum([torch.norm(p.grad).cpu().item()**2 for p in model.parameters() if p.grad is not None ]))

  return sum_loss/counter, sum_class_loss/counter, sum_pos_loss/counter, sum_b_acc/counter, sum_acc/counter, grad_norm  

def load_current_state(cfg, model):
  current_epoch = 0
  history = {"train_loss":[], "train_class_loss":[], "train_pos_loss": [], "train_acc":[], "train_b_acc":[], "train_grad_norm": [], "val_loss":[], "val_class_loss":[], "val_pos_loss": [], "val_acc":[], "val_b_acc":[], "val_grad_norm": [], "best_epoch": None}  
  current_model = glob.glob(os.path.join(cfg.OUTPUT.LOCATION, 'current_model_epoch*.pt'))

  if len(current_model) > 0:
    current_model.sort()
    current_epoch = current_model[-1].split('current_model_epoch')
    weights = torch.load(current_model[-1])
    model.load_state_dict(model.update_version(weights))

    current_model = current_model[-1].split('current_model_epoch')
    current_model = current_model[-1].split('.')
    current_epoch = int(current_model[0])

    hist_file = os.path.join(cfg.OUTPUT.LOCATION, "history.json")

    if os.path.isfile(hist_file):
      hist_file = open(hist_file, "r")
      history   = json.load(hist_file)

  return model, current_epoch + 1, history

def save_current_state(cfg, model, history, epoch): 
  model_pattern = os.path.join(cfg.OUTPUT.LOCATION, 'current_model_epoch{:02d}.pt')  
  torch.save(model.state_dict(), model_pattern.format(epoch))

  hist_file = open(os.path.join(cfg.OUTPUT.LOCATION, "history.json"), "w")
  hist_file.write(json.dumps(history))

  previous_model = model_pattern.format(epoch - 1)

  if epoch > 1 and os.path.isfile(previous_model):
    os.remove(previous_model)

def train(train_loader, val_loader, cfg):
    # Instantiating the models
    model, device = build_model(cfg)
    best_model = os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt')

    # Defining loss function and optimizer
    class_weight = None
  
    if cfg.TRAIN.USE_CLASS_WEIGHT:
      class_weight = np.sum(train_loader.dataset.class_histogram) / np.array(train_loader.dataset.class_histogram) ## 1 / freq%
      class_weight = class_weight / np.sum(class_weight) ## norm in [0, 1]
      print("Cross-entropy weights", class_weight)
      class_weight = torch.FloatTensor(class_weight).to(device)

    criterion = nn.CrossEntropyLoss(weight = class_weight)
    criterion_t = nn.MSELoss()
    scheduler = None

    if cfg.TRAIN.OPT == "adam":
      optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR, weight_decay = cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPT == "sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum = cfg.TRAIN.MOMENTUM, weight_decay = cfg.TRAIN.WEIGHT_DECAY)
    elif cfg.TRAIN.OPT == "rmsprop":
      optimizer = torch.optim.RMSprop(model.parameters(), lr=cfg.TRAIN.LR, weight_decay = cfg.TRAIN.WEIGHT_DECAY)

    if cfg.TRAIN.SCHEDULER == "step":
      scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 5)
    elif cfg.TRAIN.SCHEDULER == "exp":  
      scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    
    data_features = ""

    if cfg.MODEL.USE_ACTION:
      data_features = "action"
    if cfg.MODEL.USE_OBJECTS:
      data_features +=  "image" if data_features == "" else "+image"
    if cfg.MODEL.USE_AUDIO:
      data_features +=  "sound" if data_features == "" else "+sound"

    aug_data = "raw data"

    if cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS or cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS:
      aug_data = "aug"
      if cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS:
        aug_data += " img"
      if cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS:
        aug_data += " time"          

    print("Training of step recognition for {}: model {} - optimizer {} - features {} - {} ".format(cfg.SKILLS[0]["NAME"], model.__class__.__name__, optimizer.__class__.__name__, data_features, aug_data ))

    model, first_epoch, history = load_current_state(cfg, model)
    progress = tqdm.tqdm(total = len(train_loader), unit= "step", bar_format='{desc}|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} - {rate_fmt}]{postfix}' )

    best_val_loss = float('inf') if len(history["val_loss"]) == 0 else np.min(history["val_loss"])
    best_val_acc  = float('inf') if len(history["val_acc"]) == 0 else np.min(history["val_acc"])

    for epoch in range(first_epoch, cfg.TRAIN.EPOCHS + 1):
      progress.set_description("Epoch {}/{} ".format(epoch, cfg.TRAIN.EPOCHS))
      progress.reset()
      
      train_loss, train_class_loss, train_pos_loss, train_b_acc, train_acc, grad_norm = train_step(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg, progress=progress)
      history["train_loss"].append(train_loss)
      history["train_class_loss"].append(train_class_loss)
      history["train_pos_loss"].append(train_pos_loss)
      history["train_acc"].append(train_acc)
      history["train_b_acc"].append(train_b_acc)      
      history["train_grad_norm"].append(grad_norm)

      with torch.no_grad():
        val_loss, val_class_loss, val_pos_loss, val_b_acc, val_acc, grad_norm = train_step(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg, progress=progress)
        
      history["val_loss"].append(val_loss)
      history["val_class_loss"].append(val_class_loss)
      history["val_pos_loss"].append(val_pos_loss)
      history["val_acc"].append(val_acc)
      history["val_b_acc"].append(val_b_acc)      
      history["val_grad_norm"].append(grad_norm)

      progress.set_postfix({"Cross entropy": train_class_loss, "MSE": train_pos_loss, "Total loss": train_loss, "acc": train_acc, 
                            "val Cross entropy": val_class_loss, "val MSE": val_pos_loss, "val Total loss": val_loss, "val acc": val_acc})
      print("")

      if scheduler is not None:
        scheduler.step()
        print("Learning rate: ", scheduler.get_last_lr())
      if val_loss < best_val_loss:
        history["best_epoch"] = epoch
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model)

      save_current_state(cfg, model, history, epoch) 

    plot_history(history, cfg)        

    if cfg.TRAIN.RETURN_METRICS:
      return best_model, best_val_loss, best_val_acc
    else:
      return best_model

@torch.no_grad()
def evaluate(model, data_loader, cfg, aggregate_avg = False):
  number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.eval()  
  
  outputs = []
  targets = []

  for action, obj, frame, audio, label, _, _, frame_idx, videos in data_loader:
    h = model.init_hidden(len(action))

    out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)
    out    = torch.softmax(out[..., :number_classes], dim = -1).cpu().detach().numpy()
    label  = label.cpu().numpy()
    frame_idx = frame_idx.cpu().numpy()

    for video_id, video_frames, frame_target, frame_pred in zip(videos, frame_idx, label, out):
      aux_frame = []
      aux_targets = []
      aux_outputs = []

      for frame, target, pred in zip(video_frames, frame_target, frame_pred):
        if frame > 0: #it's equal to test the mask value, like in train_step
          aux_frame.append(frame)
          aux_targets.append(target)
          aux_outputs.append(pred)

          targets.append(target)
          outputs.append(pred)

      save_video_evaluation(video_id[0], aux_frame, aux_targets, aux_outputs, cfg)  

  targets = np.array(targets)
  outputs = np.array(outputs)

  classes = [ i for i in range(number_classes)]
  classes_desc = [ "Step " + str(i + 1)  for i in range(number_classes)]
  classes_desc[-1] = "No step"
  save_evaluation(targets, np.argmax(outputs, axis = 1), classes, cfg, label_order = classes_desc)

@torch.no_grad()
def extract_features(model, data_loader, cfg, aggregate_avg = False):
  number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.eval()  

  # assign vocabulary
  skill_steps = np.array([
      step
      for skill in cfg.SKILLS
      for step in skill['STEPS']
  ])

  for idx, (action, obj, frame, audio, label, _, _, frame_idx, videos) in enumerate(data_loader, start=1):
    print("|- Batch", idx)
    h = model.init_hidden(len(action))

    out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)
    out    = torch.softmax(out[..., :number_classes], dim = -1).cpu().detach().numpy()
    label  = label.cpu().numpy()
    frame_idx = frame_idx.cpu().numpy()

    for video_id, video_frames, frame_target, frame_pred, a_feat, o_feat, f_feat, s_feat in zip(videos, frame_idx, label, out, action, obj, frame, audio):
      print("|--", video_id[0])
      frames  = []
      outputs = []
      output_desc = []
      targets = []
      target_desc = []
      action_feature = []
      frame_feature = []
      obj_feature = []
      sound_feature = []      

      for frame, target, pred, a, o, f, s in zip(video_frames, frame_target, frame_pred, a_feat, o_feat, f_feat, s_feat):
        if frame > 0:
          frames.append(frame)

          targets.append(target)
          target_desc.append("No step" if target >= len(skill_steps) else skill_steps[target])

          outputs.append(pred)
          step_idx  = np.argmax(pred)
          output_desc.append("No step" if step_idx >= len(skill_steps) else skill_steps[step_idx])

          action_feature.append(a.numpy())
          frame_feature.append(f.numpy())
          obj_feature.append(o.numpy())
          sound_feature.append(s.numpy())

      np.savez(os.path.join(cfg.OUTPUT.LOCATION, "{}-features.npz".format(video_id[0])), action=np.array(action_feature), frame=np.array(frame_feature), object=np.array(obj_feature), sound=np.array(sound_feature))
      np.savez(os.path.join(cfg.OUTPUT.LOCATION, "{}-window_label.npz".format(video_id[0])), frame_idx=np.array(frames), label=np.array(targets), label_desc=np.array(target_desc), label_pred=np.array(outputs), label_pred_desc=np.array(output_desc))
      save_video_evaluation(video_id[0], frames, targets, outputs, cfg)

def plot_history(history, cfg):
  hist_file = open(os.path.join(cfg.OUTPUT.LOCATION, "history.json"), "w")
  hist_file.write(json.dumps(history))

  figure = plt.figure(figsize = (1280 / 100, 480 / 100), dpi = 100)  
  #====================================================================================================================#
  plt.subplot(2, 3, 1)   
  plot_data(history["train_class_loss"], history["val_class_loss"], ylabel = "Cross Entropy", mark_best = history["best_epoch"])

  plt.subplot(2, 3, 2)   
  plot_data(history["train_pos_loss"], history["val_pos_loss"], ylabel = "MSE", mark_best = history["best_epoch"])  

  plt.subplot(2, 3, 3)   
  plot_data(history["train_loss"], history["val_loss"], ylabel = "Total Loss", mark_best = history["best_epoch"])    

  plt.subplot(2, 3, 4)    
  plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Categorical accuracy", mark_best = history["best_epoch"])     

##  plt.subplot(2, 3, 5)    
##  plot_data(history["train_b_acc"], history["val_b_acc"], xlabel = "Epoch", ylabel = "Balanced accuracy", mark_best = history["best_epoch"])       

  plt.subplot(2, 3, 5)    
  plot_data(history["train_grad_norm"], [], xlabel = "Epoch", ylabel = "Gradient norm", mark_best = history["best_epoch"])       

  figure.tight_layout()
  figure.savefig(os.path.join(cfg.OUTPUT.LOCATION, "history_chart.png"))

  summary_history(history, cfg)

def summary_history(history, cfg=None):
  path = history

  if isinstance(history, str):
    history = open(os.path.join(path, "history.json"), "r")
    history = json.load(history)
  elif isinstance(history, dict):
    path = cfg.OUTPUT.LOCATION

  figure = plt.figure(figsize = (1024 / 100, 290 / 100), dpi = 100)
  #====================================================================================================================#  
  plt.subplot(1, 2, 1)   
  plot_data(history["train_loss"], history["val_loss"], xlabel = "Epoch", ylabel = "Total Loss", mark_best = history["best_epoch"])    

  plt.subplot(1, 2, 2)   
  plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Categorical accuracy", mark_best = history["best_epoch"])     

  figure.tight_layout()
  figure.savefig(os.path.join(path, "history_chart_short.png"))
  #====================================================================================================================#  

def plot_data(train, val, xlabel = None, ylabel = None, mark_best = None, palette = {"blue": "#1F77B4", "red": "#B41D1EFF", "grey": "#808080"}):
  plt.plot(range(1, len(train) + 1), train, color = palette["blue"], linestyle="-")

  if len(val) == 0:
    plt.legend(['Training'])
  else: 
    last_index = len(train) - 1
    diff  = abs(train[last_index] - val[last_index])  

    plt.plot(range(1, len(val) + 1), val, color = palette["red"], linestyle='--')
    plt.plot(1, np.min([train, val]), 'white')

    plt.legend(['Training', 'Validation', 'dif. {:.4f}'.format(diff)])

  if xlabel is not None:
    plt.xlabel(xlabel)
  if ylabel is not None:
    plt.ylabel(ylabel)    
  if mark_best is not None:  
    plt.axvline(x = mark_best, color = palette["grey"])

def save_evaluation(expected, predicted, classes, cfg, label_order = None, normalize = "true", file_name = "confusion_matrix.png", pad = None):
  number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1
  file = open(os.path.join(cfg.OUTPUT.LOCATION, "metrics.txt"), "w")

  try:
    file.write(classification_report(expected, predicted, zero_division = 0, labels = classes, target_names = label_order)) 
    file.write("\n\n")     
    file.write("Categorical accuracy: {:.2f}\n".format(accuracy_score(expected, predicted)))
    file.write("Weighted accuracy: {:.2f}\n".format(weighted_accuracy(expected, predicted, number_classes)))
    file.write("Balanced accuracy: {:.2f}\n".format(my_balanced_accuracy_score(expected, predicted)))
  finally:
    file.close() 
  #========================================================================================================================#
  cm = confusion_matrix(expected, predicted, normalize = normalize, labels = classes)

  if pad is not None:
    cm = np.pad(cm, pad)

  df = pd.DataFrame(cm, columns = label_order, index = label_order)
  df.index.name   = 'Expected'
  df.columns.name = None    

  figure = plt.figure(figsize = (1366 / 100, 768 / 100), dpi = 100)

  try:
    sb.set(font_scale = 1.4)#for label size
    ax = plt.axes()
    ax.set_title('Predicted', fontsize = 16)  
    sb.heatmap(df, ax = ax, cmap = "Blues", annot = True, fmt = '.2f', annot_kws=None if df.shape[0] < 20 else {"size": 6}, vmin = 0.0, vmax = 1.0)# font size  linewidths
    figure.tight_layout()
    figure.savefig(os.path.join(cfg.OUTPUT.LOCATION, file_name))    
  finally:
    plt.close()
    sb.reset_orig()

def save_video_evaluation(video_id, window_last_frame, expected, probs, cfg):
  expected = np.array(expected)
  output_location = os.path.join(cfg.OUTPUT.LOCATION, "video_evaluation")

  if not os.path.isdir(output_location):
    os.mkdir(output_location)

  last_expected = np.max(expected)
  expected[expected == last_expected] = -1

  predicted = np.argmax(probs, axis = 1) 
  last_predicted = np.max(predicted)
  predicted[predicted == last_predicted] = -1

  classes_desc = [ "No step" if i == 0 else "Step " + str(i)  for i in range(last_expected + 1)]  

  figure = plt.figure(figsize = (1024 / 100, 768 / 100), dpi = 100)

  plt.subplot(2, 1, 1)
  plt.step(window_last_frame, expected, c="royalblue")
  plt.yticks( [ i - 1 for i in range(last_expected + 1) ], classes_desc)  

  plt.step(window_last_frame, predicted, c="orange")
  plt.yticks( [ i - 1 for i in range(last_predicted + 1) ], classes_desc)    

  plt.legend(["target", "predicted"])
  plt.grid(axis = "y")      

  probs = np.max(probs, axis = 1)

  plt.subplot(2, 1, 2)
  plt.plot(window_last_frame, probs, marker = 'o', c="teal")
  plt.axhline(y = np.mean(probs), color = 'grey', linestyle = ':')  
  plt.ylabel("Confidence")
  plt.xlabel("last window frame")  

  figure.tight_layout()
  figure.savefig(os.path.join(output_location, "{}-step_variation.png".format(video_id)))    


##https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/metrics/_classification.py
##treating division-by-zero
def my_balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight)
    ## with np.errstate(divide="ignore", invalid="ignore"):
    ##     per_class = np.diag(C) / C.sum(axis=1)
    per_class = np.divide(np.diag(C), C.sum(axis=1), out=np.zeros_like(np.diag(C), dtype='float64'), where=C.sum(axis=1)!=0)
    ## if np.any(np.isnan(per_class)):
    ##     warnings.warn("y_pred contains classes not in y_true")
    ##     per_class = per_class[~np.isnan(per_class)]
    score = np.mean(per_class)
    if adjusted:
        n_classes = len(per_class)
        chance = 1 / n_classes
        score -= chance
        score /= 1 - chance
    return score  

def weighted_accuracy(y_true, y_pred, number_classes):
  sample_weight = np.ones(y_true.shape)

  for cl in range(number_classes):
    count = np.sum(y_true == cl)
    sample_weight[y_true == cl] /= count  

  return accuracy_score(y_true, y_pred, sample_weight = np.array(sample_weight))    

