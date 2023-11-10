from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb
import json
from sklearn.metrics import confusion_matrix, classification_report, balanced_accuracy_score
from matplotlib import pyplot as plt
import seaborn as sb, pandas as pd
import warnings

def build_model(cfg):
  device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
  model  = OmniGRU(cfg)
  model.to(device)

  return model


def train_aux(epoch, model, criterion, criterion_t, optimizer, loader, is_training, device, cfg):
  if is_training:
    model.train()
    h = model.init_hidden(cfg.TRAIN.BATCH_SIZE)
  else:
    model.eval()  

  avg_loss = 0.
  avg_acc = 0.
  counter = 0
  count_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1

  for action, obj, frame, audio, label,label_t,mask,_,_ in loader:
    label = nn.functional.one_hot(label,count_classes)
    counter += 1

    if not is_training:
      h = model.init_hidden(len(action))

    h = torch.zeros_like(h)
    model.zero_grad()

    out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
    out_t = torch.softmax(out[..., None, count_classes:],dim=-1)
    out = out[..., None, :count_classes]    
    out_masked = out*mask.to(device)
    label_masked = mask.to(device)*label.to(device).float()
    loss = criterion(out_masked, label_masked)+criterion_t(out_t*mask.to(device), mask.to(device)*label_t.to(device).float())

    if is_training:
      loss.backward()
      optimizer.step()

    avg_loss += loss.item()
    out_masked = torch.argmax(out_masked, axis = 2)
    label_masked = torch.argmax(label_masked, axis = 2)
    avg_acc += my_balanced_accuracy_score(np.concatenate(label_masked.cpu().numpy()), np.concatenate(out_masked.cpu().numpy()), labels = range(count_classes))

    if counter % 1 == 0:
        print("|-- Epoch {}/{} - step: {}/{} ({}) - avg. Loss: {} - avg. Balanced accuracy {}".format(epoch, cfg.TRAIN.EPOCHS, counter, len(loader), "Training"if is_training else "Validation", avg_loss/counter, avg_acc/counter))     

  return avg_loss, avg_acc


def train(train_loader, val_loader, cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Instantiating the models
    model = build_model(cfg)
    
    # Defining loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    criterion_t = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    
    print("Starting Training of OmniGRU model for step recognition")
    # Start training loop
    best_val_loss = float('inf')

    history = {"train_loss":[], "val_loss":[], "train_acc":[], "val_acc":[], "best_epoch": None}

    for epoch in range(1,cfg.TRAIN.EPOCHS+1):
        avg_loss, avg_acc = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg)
        train_loss = avg_loss/len(train_loader)
        train_acc = avg_acc/len(train_loader)
        print("|- Epoch {}/{} - ({}) avg. Loss: {} - avg. Balanced accuracy {}".format(epoch, cfg.TRAIN.EPOCHS, "Training", train_loss, train_acc))     
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        avg_loss, avg_acc = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg)
        val_loss = avg_loss/len(val_loader)
        val_acc  = avg_acc/len(val_loader)

        print("|- Epoch {}/{} - ({}) avg. Loss: {} - avg. Balanced accuracy {}".format(epoch, cfg.TRAIN.EPOCHS, "Validation", val_loss, val_acc))     
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        if val_loss < best_val_loss:
            history["best_epoch"] = epoch
            best_val_loss = val_loss
            print("|--- Found new best validation loss (saving model) in the epoch {}".format(epoch))     
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
    count_classes = cfg.MODEL.OUTPUT_DIM if cfg.MODEL.APPEND_OUT_POSITIONS == 2 else cfg.MODEL.OUTPUT_DIM + 1
    summary = {}

    for _, (action, obj, frame, audio, label, _, mask, frame_idx, id) in tqdm.tqdm(enumerate(data_loader)):
      h = model.init_hidden(action.shape[0])
      out, _ = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
      out_aux = (out[..., None, :count_classes]*mask.to(device)).cpu().detach().numpy()

      ##the same frame_idx/video could be returned in different iterations.
      ##accumulate this informations in a same structure
      for video, frame, output in zip(id, frame_idx, out_aux):
        video = video[0]

        if not video in summary:
          summary[video] = {"frame_idx": {}, "label": {} }

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
        #maximum confidence of each row (window)
        out_aux = np.max(summary[v]["frame_idx"][f], axis = 1) 
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

    classes = [ i for i in range(count_classes)]
    classes_desc = [ "Step " + str(i + 1)  for i in range(count_classes)]
    classes_desc[-1] = "No step"
    save_evaluation(targets, np.argmax(outputs, axis = 1), classes, cfg, label_order = classes_desc)

    np.save(f'{cfg.OUTPUT.LOCATION}/video_ids.npy', video_ids)
    np.save(f'{cfg.OUTPUT.LOCATION}/frames.npy', frames)
    np.save(f'{cfg.OUTPUT.LOCATION}/outputs.npy', outputs)
    np.save(f'{cfg.OUTPUT.LOCATION}/targets.npy', targets)

def plot_history(history, cfg):
  hist_file = open(os.path.join(cfg.OUTPUT.LOCATION, "history.json"), "w")
  hist_file.write(json.dumps(history))

  figure = plt.figure()      
  #====================================================================================================================#
  plt.subplot(2, 1, 1)    
  plot_data(history["train_loss"], history["val_loss"], ylabel = "Loss", mark_best = history["best_epoch"])

  plt.subplot(2, 1, 2)    
  plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Balanced accuracy", mark_best = history["best_epoch"])   

  figure.tight_layout()
  figure.savefig(os.path.join(cfg.OUTPUT.LOCATION, "history_chart.png"))  

def plot_data(train, val, xlabel = None, ylabel = None, mark_best = None):
  if len(train) > 0 and len(val) > 0:
    last_index = len(train) - 1
    diff  = abs(train[last_index] - val[last_index])  

    plt.plot(range(1, len(train) + 1), train, 'b-') 
    plt.plot(range(1, len(val) + 1), val, 'r--') 
    plt.plot(1, np.min([train, val]), 'white')

    plt.legend(['Training', 'Validation', 'dif. {:.4f}'.format(diff)])

    if xlabel is not None:
      plt.xlabel(xlabel)
    if ylabel is not None:
      plt.ylabel(ylabel)    
    if mark_best is not None:  
      plt.axvline(x = mark_best, color = 'grey')

def save_evaluation(expected, predicted, classes, cfg, class_names = None, label_order = None, normalize = "true", file_name = "confusion_matrix.png", pad = None):
  file = open(os.path.join(cfg.OUTPUT.LOCATION, "metrics.txt"), "w")

  try:
    file.write(classification_report(expected, predicted, zero_division = 0, labels = classes, target_names = label_order)) 
    file.write("\n\n")     
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

