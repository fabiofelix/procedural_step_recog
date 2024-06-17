from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb, ipdb
import json
import scipy
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import seaborn as sb, pandas as pd
import warnings
import glob

def build_model(cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model  = OmniGRU(cfg)
  model.to(device)

  return model, device  

def get_class_weight(class_histogram):
  class_weight = np.array(class_histogram) / np.sum(class_histogram)
  class_weight = np.divide(1.0, class_weight, where = class_weight != 0)  #avoid zero-division

  return class_weight / np.sum(class_weight) ## norm in [0, 1]  

def build_losses(loader, cfg, device):
  class_weight = None
  class_weight_tensor = None

  if cfg.TRAIN.USE_CLASS_WEIGHT:
    class_weight = get_class_weight(loader.dataset.class_histogram)
    print("|- Class weights", class_weight)

    class_weight_tensor = torch.FloatTensor(class_weight).to(device)

  return nn.CrossEntropyLoss(weight = class_weight_tensor), nn.MSELoss(), class_weight

def build_optimizer(model, cfg):
  optimizer = None
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

  return optimizer, scheduler

def train_step(model, criterion, criterion_t, optimizer, loader, is_training, device, cfg, progress, class_weight):
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
  label_expected = []
  label_predicted = []

  for counter, (action, obj, frame, audio, label, label_t, mask, frame_idx, video_id) in enumerate(loader, 1):
    label = nn.functional.one_hot(label, model.number_classes)

    if not is_training:
      h = model.init_hidden(len(label))

    h = torch.zeros_like(h)
    optimizer.zero_grad()

    out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)

    mask    = mask.to(out.device)  
    label   = label.to(out.device)
    label_t = label_t.to(out.device)

    out_t  = torch.softmax(out[..., model.number_classes:], dim = -1)  #regression of time positions. Limits the results to [0, 1] range
    out    = out[..., :model.number_classes]                           #classification of steps. CrossEntropyLoss consumes logits          

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

    ## Moving things from GPU to CPU memory and cleaning GPU to save memory
    label   = label.cpu()
    label_t = label_t.cpu()
    out     = out.cpu()
    out_t   = out_t.cpu()
    mask    = mask.cpu()
    out_masked   = out_masked.cpu()
    label_masked = label_masked.cpu()
    class_loss   = class_loss.detach().cpu()
    pos_loss     = pos_loss.detach().cpu()
    loss         = loss.detach().cpu()
    torch.cuda.empty_cache()

    sum_class_loss += class_loss.item()
    sum_pos_loss   += pos_loss.item()
    sum_loss       += loss.item()

    out_masked   = torch.argmax(torch.softmax(out_masked, dim = -1), axis = -1)
    label_masked = torch.argmax(label_masked, axis = -1)

    mask         = torch.flatten(mask).numpy()  
    label_masked = torch.flatten(label_masked).numpy()  
    out_masked   = torch.flatten(out_masked).numpy()

    label_masked_aux = []
    out_masked_aux = []

    for m, l, o in zip(mask, label_masked, out_masked):
      if m > 0:
        label_masked_aux.append(l)
        out_masked_aux.append(o)

    label_expected.extend(label_masked_aux)
    label_predicted.extend(out_masked_aux)
    label_masked_aux = np.array(label_masked_aux)
    out_masked_aux = np.array(out_masked_aux)
    sum_acc   += accuracy_score(label_masked_aux, out_masked_aux)      
    sum_b_acc += weighted_accuracy(label_masked_aux, out_masked_aux, class_weight)    

    if is_training:
      progress.update(1)
      accuracy_desc = "{}acc".format("weighted " if cfg.TRAIN.USE_CLASS_WEIGHT else ""  )
      acc_avg = (sum_b_acc if cfg.TRAIN.USE_CLASS_WEIGHT else sum_acc  )/counter
      progress.set_postfix({"Cross entropy": sum_class_loss/counter, 
                            "MSE": sum_pos_loss/counter, 
                            "Total loss": sum_loss/counter, 
                            accuracy_desc: acc_avg})

  grad_norm = np.sqrt(np.sum([torch.norm(p.grad).cpu().item()**2 for p in model.parameters() if p.grad is not None ]))

  if is_training:
    return sum_loss/counter, sum_class_loss/counter, sum_pos_loss/counter, sum_b_acc/counter, sum_acc/counter, grad_norm  
  else:
    return sum_loss/counter, sum_class_loss/counter, sum_pos_loss/counter, sum_b_acc/counter, sum_acc/counter, grad_norm, np.array(label_expected), np.array(label_predicted) 

def load_current_state(cfg, model):
  current_epoch = 0
  history = {"train_loss":[], "train_class_loss":[], "train_pos_loss": [], "train_acc":[], "train_b_acc":[], "train_grad_norm": [], 
             "val_loss":[], "val_class_loss":[], "val_pos_loss": [], "val_acc":[], "val_b_acc":[], "val_grad_norm": [], 
             "best_epoch": None}  
  current_model = glob.glob(os.path.join(cfg.OUTPUT.LOCATION, 'current_model_epoch*.pt'))

  if len(current_model) > 0:
    current_model.sort()
    weights = torch.load(current_model[-1])
    model.load_state_dict(model.update_version(weights))

    current_model = current_model[-1].split('current_model_epoch') #[ <path>, '20.pt' ]
    current_model = current_model[-1].split('.') #[ '20', 'pt' ]
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
    
def logging(model, optimizer, scheduler, cfg):
  data_features = ""
  aug_data = "raw data"
  schedule_msg = " "

  if cfg.MODEL.USE_ACTION:
    data_features = "action"
  if cfg.MODEL.USE_OBJECTS:
    data_features +=  "image" if data_features == "" else "+image"
  if cfg.MODEL.USE_AUDIO:
    data_features +=  "sound" if data_features == "" else "+sound"
  if cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS or cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS:
    aug_data = "aug"
    if cfg.DATASET.INCLUDE_IMAGE_AUGMENTATIONS:
      aug_data += " img"
    if cfg.DATASET.INCLUDE_TIME_AUGMENTATIONS:
      aug_data += " time"
  if scheduler is not None:
    schedule_msg = " - scheduler {} ".format(scheduler.__class__.__name__) 

  trainable_params = 0
  non_trainable_params = 0
  total_params = 0

  for param in model.parameters():
    total_params += param.numel()

    if param.requires_grad:
      trainable_params += param.numel()
    else:
      non_trainable_params += param.numel()

  print("|- Training of step recognition for {}: model {} - optimizer {}{}- features {} - {} ".format(cfg.SKILLS[0]["NAME"], model.__class__.__name__, optimizer.__class__.__name__, schedule_msg, data_features, aug_data ))
  print("{:24s}".format("|- Trainable params:"), "{:,}".format(trainable_params))
  print("{:24s}".format("|- Non-trainable params:"), "{:,}".format(non_trainable_params))
  print("{:24s}".format("|- Total params:"), "{:,}".format(total_params))

def train(train_loader, val_loader, cfg):
    best_model_path = os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt')
    model, device = build_model(cfg)
    criterion, criterion_t, train_class_weight = build_losses(train_loader, cfg, device)
    _, _, val_class_weight = build_losses(val_loader, cfg, device)
    optimizer, scheduler = build_optimizer(model, cfg)
    logging(model, optimizer, scheduler, cfg)

    model, first_epoch, history = load_current_state(cfg, model)
    progress = tqdm.tqdm(total = len(train_loader), unit= "step", bar_format='{desc}|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} - {rate_fmt}]{postfix}' )

    best_val_loss = float('inf') if len(history["val_loss"]) == 0 else np.min(history["val_loss"])
    best_val_acc  = float('inf') if len(history["val_acc"]) == 0 else history["val_acc"][np.argmin(history["val_loss"])]

    for epoch in range(first_epoch, cfg.TRAIN.EPOCHS + 1):
      progress.set_description("Epoch {}/{} ".format(epoch, cfg.TRAIN.EPOCHS))
      progress.reset()
      
      train_loss, train_class_loss, train_pos_loss, train_b_acc, train_acc, grad_norm = train_step(model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg, progress=progress, class_weight=train_class_weight)
      history["train_loss"].append(train_loss)
      history["train_class_loss"].append(train_class_loss)
      history["train_pos_loss"].append(train_pos_loss)
      history["train_acc"].append(train_acc)
      history["train_b_acc"].append(train_b_acc)      
      history["train_grad_norm"].append(grad_norm)

      with torch.no_grad():
        val_loss, val_class_loss, val_pos_loss, val_b_acc, val_acc, grad_norm, val_targets, val_outputs = train_step(model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg, progress=progress, class_weight=val_class_weight)
        
      history["val_loss"].append(val_loss)
      history["val_class_loss"].append(val_class_loss)
      history["val_pos_loss"].append(val_pos_loss)
      history["val_acc"].append(val_acc)
      history["val_b_acc"].append(val_b_acc)      
      history["val_grad_norm"].append(grad_norm)

      accuracy_desc = "{}acc".format("weighted " if cfg.TRAIN.USE_CLASS_WEIGHT else ""  )
      progress.set_postfix({"Cross entropy": train_class_loss, "MSE": train_pos_loss, "Total loss": train_loss, accuracy_desc: (train_b_acc if cfg.TRAIN.USE_CLASS_WEIGHT else train_acc), 
                            "val Cross entropy": val_class_loss, "val MSE": val_pos_loss, "val Total loss": val_loss, "val {}".format(accuracy_desc): (val_b_acc if cfg.TRAIN.USE_CLASS_WEIGHT else val_acc)})
      print("")

      if scheduler is not None:
        scheduler.step()
        print("|- Learning rate: ", scheduler.get_last_lr())
      if val_loss < best_val_loss:
        history["best_epoch"] = epoch
        best_val_loss = val_loss
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_model_path)

        ##Saving validation metrics
        classes = [ i for i in range(model.number_classes)]
        classes_desc = [ "Step " + str(i + 1)  for i in range(model.number_classes)]
        classes_desc[-1] = "No step"
        original_output = cfg.OUTPUT.LOCATION
        cfg.OUTPUT.LOCATION = os.path.join(original_output, "validation" )
        save_evaluation(val_targets, val_outputs, classes, cfg, label_order = classes_desc, class_weight = val_class_weight)
        cfg.OUTPUT.LOCATION = original_output        

      save_current_state(cfg, model, history, epoch) 

    plot_history(history, cfg)        

    if cfg.TRAIN.RETURN_METRICS:
      return best_model_path, best_val_loss, best_val_acc
    else:
      return best_model_path

@torch.no_grad()
def evaluate(model, data_loader, cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.eval()  
  
  outputs = []
  targets = []
  _, _, class_weight =  build_losses(data_loader, cfg, device)

  for action, obj, frame, audio, label, _, mask, frame_idx, videos in data_loader:
    h = model.init_hidden(len(action))

    out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)
    out    = torch.softmax(out[..., :model.number_classes], dim = -1).cpu().detach().numpy()
    label  = label.cpu().numpy()
    frame_idx = frame_idx.cpu().numpy()
    torch.cuda.empty_cache()

    for video_id, video_frames, frame_target, frame_pred, video_masks in zip(videos, frame_idx, label, out, mask):
      aux_frame = []
      aux_targets = []
      aux_outputs = []

      for frame, target, pred, mask in zip(video_frames, frame_target, frame_pred, video_masks):
        if mask > 0: #it's equal to test the mask value, like in train_step
          aux_frame.append(frame)
          aux_targets.append(target)
          aux_outputs.append(pred)

          targets.append(target)
          outputs.append(pred)

      save_video_evaluation(video_id, aux_frame, aux_targets, aux_outputs, cfg)

  targets = np.array(targets)
  outputs = np.array(outputs)

  classes = [ i for i in range(model.number_classes)]
  classes_desc = [ "Step " + str(i + 1)  for i in range(model.number_classes)]
  classes_desc[-1] = "No step"
  save_evaluation(targets, np.argmax(outputs, axis = 1), classes, cfg, label_order = classes_desc, class_weight = class_weight)

action_projection = None
obj_projection = None
frame_projection = None
img_combination_projection = None
img_projection = None
sound_projection = None
gru_input = None
gru_output = None

def act_hook(module, input, output):
  global action_projection
  action_projection = output.cpu().detach().numpy() 

def obj_hook(module, input, output):
  global obj_projection
  obj_projection = output.cpu().detach().numpy() 

def frame_hook(module, input, output):
  global frame_projection
  frame_projection = output.cpu().detach().numpy()   

def img_hook(module, input, output):
  global img_combination_projection
  global img_projection
  img_combination_projection = input[0].cpu().detach().numpy()
  img_projection = output.cpu().detach().numpy() 

def sound_hook(module, input, output):
  global sound_projection
  sound_projection = output.cpu().detach().numpy()  

def gru_hook(module, input, output):
  global gru_input
  global gru_output
  gru_input  = input[0].cpu().detach().numpy()
  gru_output = output[0].cpu().detach().numpy()    

@torch.no_grad()
def extract_features(model, data_loader, cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model.eval()  

  # assign vocabulary
  skill_steps = np.array([
      step
      for skill in cfg.SKILLS
      for step in skill['STEPS']
  ])

  layer = model._modules.get("action_fc")
  if layer is not None:
    layer.register_forward_hook(act_hook)

  layer = model._modules.get("obj_proj")
  if layer is not None:
    layer.register_forward_hook(obj_hook)  

  layer = model._modules.get("frame_proj")
  if layer is not None:
    layer.register_forward_hook(frame_hook)      

  layer = model._modules.get("obj_fc")
  if layer is not None:
    layer.register_forward_hook(img_hook)  

  layer = model._modules.get("audio_fc")
  if layer is not None:
    layer.register_forward_hook(sound_hook)    

  layer = model._modules.get("gru")
  if layer is not None:
    layer.register_forward_hook(gru_hook)  

  global action_projection
  global obj_projection  
  global frame_projection    
  global img_combination_projection
  global img_projection
  global sound_projection   
  global gru_input 
  global gru_output

  for idx, (action, obj, frame, audio, label, _, _, frame_idx, videos) in enumerate(data_loader, start=1):
    print("|- Batch", idx)
    h = model.init_hidden(len(action))

    out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float(), return_last_step = False)
    out    = torch.softmax(out[..., :model.number_classes], dim = -1).cpu().detach().numpy()
    label  = label.cpu().numpy()
    frame_idx = frame_idx.cpu().numpy()
    torch.cuda.empty_cache()

    if action_projection is None:
      action_projection = np.zeros(action.shape[:2] + (0,))
    if obj_projection is None:  
      obj_projection = np.zeros(obj.shape[:2] + (0,))       
    if frame_projection is None:  
      frame_projection = np.zeros(obj.shape[:2] + (0,))             
    if img_projection is None:  
      img_projection = np.zeros(obj.shape[:2] + (0,))       
    if img_combination_projection is None:  
      img_combination_projection = np.zeros(obj.shape[:2] + (0,)) 
    if sound_projection is None:  
      sound_projection = np.zeros(audio.shape[:2] + (0,)) 

    for video_id, video_frames, frame_target, frame_pred, a_feat, o_feat, f_feat, s_feat, act_proj, frame_proj, obj_proj, img_comb, img_proj, sound_proj, feat_concat, gru_proj in zip(videos, frame_idx, label, out, action, obj, frame, audio, action_projection, frame_projection, obj_projection, img_combination_projection, img_projection, sound_projection, gru_input, gru_output):
      print("|--", video_id)
      frames  = []
      outputs = []
      output_desc = []
      targets = []
      target_desc = []
      action_feature = []
      frame_feature = []
      obj_feature = []
      sound_feature = []
      act_proj_feature = []      
      obj_proj_feature = []            
      frame_proj_feature = []            
      img_comb_feature = []
      img_proj_feature = []      
      sound_proj_feature = []      
      features_concat = []      
      gru_feature = []

      for frame, target, pred, a, o, f, s, a_p, f_p, o_p, i_c, i_p, s_p, f_c, g_p in zip(video_frames, frame_target, frame_pred, a_feat, o_feat, f_feat, s_feat, act_proj, frame_proj, obj_proj, img_comb, img_proj, sound_proj, feat_concat, gru_proj):
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
          act_proj_feature.append(a_p)
          obj_proj_feature.append(o_p)
          frame_proj_feature.append(f_p)
          img_comb_feature.append(i_c)
          img_proj_feature.append(i_p)
          sound_proj_feature.append(s_p)
          features_concat.append(f_c)
          gru_feature.append(g_p)

      np.savez(os.path.join(cfg.OUTPUT.LOCATION, "{}-features.npz".format(video_id)), 
                            action=np.array(action_feature), frame=np.array(frame_feature), object=np.array(obj_feature), sound=np.array(sound_feature),
                            action_proj=np.array(act_proj_feature), 
                            frame_proj=np.array(frame_proj_feature), obj_proj=np.array(obj_proj_feature), img_comb=np.array(img_comb_feature), img_proj=np.array(img_proj_feature), 
                            sound_proj=np.array(sound_proj_feature), 
                            feature_concat=features_concat, gru_feature=np.array(gru_feature)
                            )
      np.savez(os.path.join(cfg.OUTPUT.LOCATION, "{}-window_label.npz".format(video_id)), frame_idx=np.array(frames), label=np.array(targets), label_desc=np.array(target_desc), label_pred=np.array(outputs), label_pred_desc=np.array(output_desc))
      save_video_evaluation(video_id, frames, targets, outputs, cfg)  

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

  if cfg.TRAIN.USE_CLASS_WEIGHT:
    plot_data(history["train_b_acc"], history["val_b_acc"], xlabel = "Epoch", ylabel = "Weighted accuracy", mark_best = history["best_epoch"])       
  else:
    plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Categorical accuracy", mark_best = history["best_epoch"])     

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

  if cfg.TRAIN.USE_CLASS_WEIGHT:
    plot_data(history["train_b_acc"], history["val_b_acc"], xlabel = "Epoch", ylabel = "Weighted accuracy", mark_best = history["best_epoch"])         
  else:
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

def save_evaluation(expected, predicted, classes, cfg, label_order = None, normalize = "true", file_name = "confusion_matrix.png", pad = None, class_weight = None):
  file = open(os.path.join(cfg.OUTPUT.LOCATION, "metrics.txt"), "w")

  try:
    file.write(classification_report(expected, predicted, zero_division = 0, labels = classes, target_names = label_order)) 
    file.write("\n\n")     
    file.write("Categorical accuracy: {:.2f}\n".format(accuracy_score(expected, predicted)))
    file.write("Weighted accuracy: {:.2f}\n".format(weighted_accuracy(expected, predicted, class_weight)))    
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
    sb.set_theme(font_scale = 1.4)#for label size
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
  accuracy  = accuracy_score(expected, predicted)
  acc_desc  = "acc"

  if cfg.TRAIN.USE_CLASS_WEIGHT:
    acc_desc     = "weighted acc"
    class_weight = get_class_weight([ np.sum(expected == c) for c in np.unique(expected) ])
    accuracy     = weighted_accuracy(expected, predicted, class_weight=class_weight)

  precision = precision_score(expected, predicted, average=None)
  recall    = recall_score(expected, predicted, average=None)

  figure = plt.figure(figsize = (1024 / 100, 768 / 100), dpi = 100)

  try:
    plt.subplot(2, 1, 1)
    plt.step(window_last_frame, expected, c="royalblue")
    plt.yticks( [ i - 1 for i in range(last_expected + 1) ], classes_desc)  

    plt.step(window_last_frame, predicted, c="orange")
    plt.yticks( [ i - 1 for i in range(last_predicted + 1) ], classes_desc)
    
    plt.plot(1, np.min([expected, predicted]), 'white')
    plt.plot(1, np.min([expected, predicted]), 'white')
    plt.plot(1, np.min([expected, predicted]), 'white')

    plt.legend(["target", "predicted", 
                "{} {:.2f}".format(acc_desc, accuracy ),
                "precision {:.2f}+/-{:.2f}".format( precision.mean(), precision.std() ),
                "recall {:.2f}+/-{:.2f}".format( recall.mean(), recall.std() )
                ])
    plt.grid(axis = "y")      

    probs = np.max(probs, axis = 1)

    plt.subplot(2, 1, 2)
    plt.plot(window_last_frame, probs, marker = 'o', c="teal")
    plt.axhline(y = np.mean(probs), color = 'grey', linestyle = ':')  
    plt.ylabel("Confidence")
    plt.xlabel("last window frame")  

    figure.tight_layout()
    figure.savefig(os.path.join(output_location, "{}-step_variation.png".format(video_id)))
  finally:
    plt.close()

def weighted_accuracy(y_true, y_pred, class_weight):
  sample_weight = np.zeros(y_true.shape)

  for cl in range(len(class_weight)):
    sample_weight[y_true == cl] = class_weight[cl]

  return accuracy_score(y_true, y_pred, sample_weight = sample_weight)

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
