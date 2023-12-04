from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb
import json
import scipy
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score, accuracy_score
from matplotlib import pyplot as plt
import seaborn as sb, pandas as pd
import warnings

def layer_summary(name, model):
  layer = model._modules.get(name)
  weights = layer._all_weights if isinstance(layer, nn.GRU) else ["weight", "bias"]

  print("==========================================")

  for weight_list in weights:
    if isinstance(weight_list, str):  
      weight_list = [weight_list]

    for weight_name in weight_list:  
      weight = getattr(layer, weight_name).detach().cpu().numpy()
      print("Layer {} {} -> min: {} - max: {} - mean: {} - std: {}".format(name, weight_name, weight.min(), weight.max(), weight.mean(), weight.std()))

def param_regularization(model, layer_name = None):
  sum_reg = []
  type_reg = "l1"

  if layer_name is None:
    for param in model.parameters():
      if type_reg == "l1":
        sum_reg.append(torch.sum(torch.abs(param)))
      else:  
        sum_reg.append(torch.sum(param**2))
  else:   
    layer = model._modules.get(layer_name)
    params = layer._all_weights if isinstance(layer, nn.GRU) else ["weight", "bias"]
    
    for param_list in params:
        if isinstance(param_list, str):  
          param_list = [param_list]
    
        for param_name in param_list:  
          param = getattr(layer, param_name)

          if type_reg == "l1":
            sum_reg.append(torch.sum(torch.abs(param)))
          else:  
            sum_reg.append(torch.sum(param**2))


  factor =  0.01
  return factor * torch.mean(torch.tensor(sum_reg))

def build_model(cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model  = OmniGRU(cfg)
  model.to(device)

  return model  

def train_step(epoch, model, criterion, criterion_t, optimizer, loader, is_training, device, cfg, progress):
  if is_training:
    model.train()
    h = model.init_hidden(cfg.TRAIN.BATCH_SIZE)
  else:
    model.eval()  

  sum_class_loss = 0.
  sum_pos_loss = 0.
  sum_loss = 0.
  sum_b_acc = 0.
  number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1

  for counter, (action, obj, frame, audio, label, label_t, mask, _, _) in enumerate(loader, 1):
    label = nn.functional.one_hot(label, number_classes)

    if not is_training:
      h = model.init_hidden(len(action))

    h = torch.zeros_like(h)
    optimizer.zero_grad()

    out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
    out_t  = torch.softmax(out[..., None, number_classes:], dim = -1)  #regression of time positions. Limits the results to [0, 1] range
    out    = out[..., None, :number_classes]                           #classification of steps. CrossEntropyLoss consumes logits    

    out_masked   = out * mask.to(device)
    label_masked = mask.to(device) * label.to(device).float()

    class_loss = criterion(out_masked, label_masked)
    pos_loss   = criterion_t(out_t * mask.to(device), mask.to(device) * label_t.to(device).float())
    loss       = class_loss + pos_loss

    if is_training:
      loss.backward()
      optimizer.step()

    sum_class_loss += class_loss.item()
    sum_pos_loss   += pos_loss.item()
    sum_loss       += loss.item()

    out_masked   = torch.argmax(out_masked, axis = 2)
    label_masked = torch.argmax(label_masked, axis = 2)
    sum_b_acc   += my_balanced_accuracy_score(np.concatenate(label_masked.cpu().numpy()), np.concatenate(out_masked.cpu().numpy()), labels = range(number_classes))

    if is_training:
      progress.update(1)
      progress.set_postfix({"Cross entropy": sum_class_loss/counter, 
                            "MSE": sum_pos_loss/counter, 
                            "Total loss": sum_loss/counter, 
                            "Balanced acc": sum_b_acc/counter})

  grad_norm = np.sqrt(np.sum([torch.norm(p.grad).cpu().item()**2 for p in model.parameters() if p.grad is not None ]))

  return sum_loss/counter, sum_class_loss/counter, sum_pos_loss/counter, sum_b_acc/counter, grad_norm


def train(train_loader, val_loader, cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiating the models
    model = build_model(cfg)
    
    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_t = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    
    print("Starting Training of OmniGRU model for step recognition")
    best_val_loss = float('inf')

    history = {"train_loss":[], "train_class_loss":[], "train_pos_loss": [], "train_acc":[], "train_grad_norm": [], "val_loss":[], "val_class_loss":[], "val_pos_loss": [], "val_acc":[], "val_grad_norm": [], "best_epoch": None}
    progress = tqdm.tqdm(total = len(train_loader), unit= "step", bar_format='{desc}|{bar:10}| {n_fmt}/{total_fmt} [{elapsed}<{remaining} - {rate_fmt}]{postfix}' )

    for epoch in range(1, cfg.TRAIN.EPOCHS + 1):
      progress.set_description("Epoch {}/{} ".format(epoch, cfg.TRAIN.EPOCHS))
      progress.reset()
      
      train_loss, train_class_loss, train_pos_loss, train_acc, grad_norm = train_step(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg, progress=progress)
      history["train_loss"].append(train_loss)
      history["train_class_loss"].append(train_class_loss)
      history["train_pos_loss"].append(train_pos_loss)
      history["train_acc"].append(train_acc)
      history["train_grad_norm"].append(grad_norm)

      val_loss, val_class_loss, val_pos_loss, val_acc, grad_norm = train_step(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg, progress=progress)
      history["val_loss"].append(val_loss)
      history["val_class_loss"].append(val_class_loss)
      history["val_pos_loss"].append(val_pos_loss)
      history["val_acc"].append(val_acc)
      history["val_grad_norm"].append(grad_norm)

      progress.set_postfix({"Cross entropy": train_class_loss, "MSE": train_pos_loss, "Total loss": train_loss, "Balanced acc": train_acc, 
                            "val Cross entropy": val_class_loss, "val MSE": val_pos_loss, "val Total loss": val_loss, "val Balanced acc": val_acc})
      print("")

      if val_loss < best_val_loss:
        history["best_epoch"] = epoch
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt'))

    plot_history(history, cfg)        
    return model

def evaluate(model, data_loader, cfg, aggregate_avg = False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    outputs = []
    targets = []
    video_ids = []
    frames = []

    number_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1
    summary = {}

    for action, obj, frame, audio, label, _, mask, frame_idx, id in tqdm.tqdm(data_loader, total = len(data_loader), desc = "Evaluation steps"):
      h = model.init_hidden(action.shape[0])
      out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
      out_aux = (out[..., None, :number_classes]*mask.to(device)).cpu().detach().numpy()

      ##the same frame_idx/video could be returned in different iterations.
      ##accumulate this informations in a same structure
      for video, frame, output in zip(id, frame_idx, out_aux):        
        video = video[0]

        if not video in summary:
          summary[video] = {"frame_idx": {}, "label": {}, "before_gru_space":{}, "after_gru_space": {} }

        for f, o, l in zip(frame, output, label):          
          f = int(f)

          if not f in summary[video]["frame_idx"]:
            summary[video]["frame_idx"][f] = []
            summary[video]["label"][f] = []

          summary[video]["frame_idx"][f].append(o)  
          summary[video]["label"][f].append(l.numpy())  


    video_ids = []
    frames = []
    outputs = []
    targets = []

    #Aggregates information of summary
    for v in summary:
      for f in summary[v]["frame_idx"]:
        video_ids.append(v)
        frames.append(f)
        probs = [ scipy.special.softmax(logit) for logit in summary[v]["frame_idx"][f] ]
        #maximum confidence of each row (window)
        out_aux = np.max(probs, axis = 1) 
        #index of the row (window) with maximum confidence
        out_aux = np.argmax(out_aux)
        #returns only the row with the maximum
        out_aux = summary[v]["frame_idx"][f][out_aux]

        if aggregate_avg:
          out_aux = np.mean(summary[v]["frame_idx"][f], axis = 0)

        outputs.append(out_aux)  
        targets.append(np.max(np.concatenate(summary[v]["label"][f])))

    video_ids = np.array(video_ids)
    frames  = np.array(frames)
    outputs = np.array(outputs)
    targets = np.array(targets)

    classes = [ i for i in range(number_classes)]
    classes_desc = [ "Step " + str(i + 1)  for i in range(number_classes)]
    classes_desc[-1] = "No step"
    save_evaluation(targets, np.argmax(outputs, axis = 1), classes, cfg, label_order = classes_desc)

    np.save(f'{cfg.OUTPUT.LOCATION}/video_ids.npy', video_ids)
    np.save(f'{cfg.OUTPUT.LOCATION}/frames.npy', frames)
    np.save(f'{cfg.OUTPUT.LOCATION}/outputs.npy', outputs)
    np.save(f'{cfg.OUTPUT.LOCATION}/targets.npy', targets)

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
  plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Balanced accuracy", mark_best = history["best_epoch"])     

  plt.subplot(2, 3, 5)    
  plot_data(history["train_grad_norm"], [], xlabel = "Epoch", ylabel = "Gradient norm", mark_best = history["best_epoch"])       

  figure.tight_layout()
  figure.savefig(os.path.join(cfg.OUTPUT.LOCATION, "history_chart.png"))  

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

def save_evaluation(expected, predicted, classes, cfg, class_names = None, label_order = None, normalize = "true", file_name = "confusion_matrix.png", pad = None):
  file = open(os.path.join(cfg.OUTPUT.LOCATION, "metrics.txt"), "w")

  try:
    file.write(classification_report(expected, predicted, zero_division = 0, labels = classes, target_names = label_order)) 
    file.write("\n\n")     
    file.write("Categorical accuracy: {:.2f}\n".format(accuracy_score(expected, predicted)))
    file.write("Balanced accuracy: {:.2f}\n".format(my_balanced_accuracy_score(expected, predicted, labels = classes)))
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

##https://github.com/scikit-learn/scikit-learn/blob/093e0cf14/sklearn/metrics/_classification.py
##Passing labels to confusion_matrix
##treating division-by-zero
def my_balanced_accuracy_score(y_true, y_pred, *, sample_weight=None, adjusted=False, labels=None):
    C = confusion_matrix(y_true, y_pred, sample_weight=sample_weight, labels=labels)
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

