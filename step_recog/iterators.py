from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb
from matplotlib import pyplot as plt

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
  counter = 0

  for action, obj, frame, audio, label,label_t,mask,_,_ in loader:
    label = nn.functional.one_hot(label,cfg.MODEL.OUTPUT_DIM)
    counter += 1

    if not is_training:
      h = model.init_hidden(len(action))

    h = torch.zeros_like(h)
    model.zero_grad()

    out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
    out_t = torch.softmax(out[..., None, cfg.MODEL.OUTPUT_DIM:],dim=-1)
    out = out[..., None, :cfg.MODEL.OUTPUT_DIM]    
    loss = criterion(out*mask.to(device), mask.to(device)*label.to(device).float())+criterion_t(out_t*mask.to(device), mask.to(device)*label_t.to(device).float())

    if is_training:
      loss.backward()
      optimizer.step()

    avg_loss += loss.item()

    if counter % 1 == 0:
      print("Epoch {}......Step: {}/{}....... Average {} Loss for Epoch: {}".format(epoch, counter, len(loader), "Training "if is_training else "Validation", avg_loss/counter))     

  return avg_loss


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
        avg_loss = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg)
        print("Epoch {}/{} Done, Total training Loss: {}".format(epoch, cfg.TRAIN.EPOCHS, avg_loss/len(train_loader)))
        history["train_loss"].append(avg_loss/len(train_loader))

        avg_loss = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg)
        val_loss = avg_loss/len(val_loader)
        print("\t\t\tEpoch {}/{} Done, Total validation Loss: {}".format(epoch, cfg.TRAIN.EPOCHS, val_loss))
        history["val_loss"].append(val_loss)
        if val_loss < best_val_loss:
            history["best_epoch"] = epoch
            best_val_loss = val_loss
            print("\t\t\tFound new best validation loss (saving model)")
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt'))

    plot_history(history, cfg)        
    return model

def evaluate(model, data_loader, cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    outputs = []
    targets = []
    video_ids = []
    frames = []
    summary = {}

    for _, (action, obj, frame, audio, label, _, mask, frame_idx, id) in tqdm.tqdm(enumerate(data_loader)):
      h = model.init_hidden(action.shape[0])
      out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())

      out_aux = (out[..., None, :cfg.MODEL.OUTPUT_DIM]*mask.to(device)).cpu().detach().numpy()

      for video, frame, output in zip(id, frame_idx, out_aux):
        video = video[0]

        if not video in summary:
          summary[video] = {"frame_idx": {}, }

        for f, o in zip(frame, output):
          f = int(f)
          if not f in summary[video]["frame_idx"]:
            summary[video]["frame_idx"][f] = []
          summary[video]["frame_idx"][f].append(o)  

      for i, _ in enumerate(frame_idx):
        video_ids.extend(np.repeat(id[i], frame_idx.shape[1]))

      frames.extend(np.concatenate(frame_idx.cpu().numpy()))
      out_aux = (out[..., None, :cfg.MODEL.OUTPUT_DIM]*mask.to(device)).cpu().detach().numpy()
      outputs.extend( out_aux )
      targets.append(np.concatenate(label.cpu().numpy()))

    video_ids = np.array(video_ids)
    frames  = np.array(frames)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)

    np.save(f'{cfg.OUTPUT.LOCATION}/video_ids.npy', video_ids)
    np.save(f'{cfg.OUTPUT.LOCATION}/frames.npy', frames)
    np.save(f'{cfg.OUTPUT.LOCATION}/outputs.npy', outputs)
    np.save(f'{cfg.OUTPUT.LOCATION}/targets.npy', targets)

def plot_history(history, cfg):
  figure = plt.figure()      
  #====================================================================================================================#
  plt.subplot(2, 1, 1)    
  plot_data(history["train_loss"], history["val_loss"], ylabel = "Loss")

  plt.subplot(2, 1, 2)    
  plot_data(history["train_acc"], history["val_acc"], xlabel = "Epoch", ylabel = "Balanced categorical acc.")   

  figure.tight_layout()
  figure.savefig(os.path.join(cfg.OUTPUT.LOCATION, "history_chart.png"))  

def plot_data(train, val, xlabel = None, ylabel = None):
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
