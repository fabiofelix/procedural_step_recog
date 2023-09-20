from step_recog.models import OmniGRU
import torch
from torch import nn
import numpy as np
import tqdm
import os
import pdb

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
    out_t = torch.softmax(out[..., cfg.MODEL.OUTPUT_DIM:],dim=-1)
    out = out[...,:cfg.MODEL.OUTPUT_DIM]
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
    for epoch in range(1,cfg.TRAIN.EPOCHS+1):
        # model.train()
        # h = model.init_hidden(cfg.TRAIN.BATCH_SIZE)
        # avg_loss = 0.
        # counter = 0
        # for action, obj, frame, audio, label,label_t,mask,_,_ in train_loader:
        #     label = nn.functional.one_hot(label,cfg.MODEL.OUTPUT_DIM)
        #     counter += 1
        #     h = torch.zeros_like(h)
        #     model.zero_grad()

        #     out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
        #     out_t = torch.softmax(out[..., cfg.MODEL.OUTPUT_DIM:],dim=-1)
        #     out = out[...,:cfg.MODEL.OUTPUT_DIM]
        #     loss = criterion(out*mask.to(device), mask.to(device)*label.to(device).float())+criterion_t(out_t*mask.to(device), mask.to(device)*label_t.to(device).float())
        #     loss.backward()
        #     optimizer.step()
        #     avg_loss += loss.item()
        #     if counter%1 == 0:
        #         print("Epoch {}......Step: {}/{}....... Average Training Loss for Epoch: {}".format(epoch, counter, len(train_loader), avg_loss/counter))
        avg_loss = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=train_loader, is_training=True, device=device, cfg=cfg)
        print("Epoch {}/{} Done, Total training Loss: {}".format(epoch, cfg.TRAIN.EPOCHS, avg_loss/len(train_loader)))

        # model.eval()
        # avg_loss = 0.
        # counter = 0
        # for action, obj, frame, audio, label,label_t,mask,_,_ in val_loader:    
        #     label = nn.functional.one_hot(label,cfg.MODEL.OUTPUT_DIM)
        #     counter += 1
        #     h = model.init_hidden(len(action))
        #     h = torch.zeros_like(h)
        #     model.zero_grad()

        #     out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())
        #     out_t = torch.softmax(out[...,cfg.MODEL.OUTPUT_DIM:],dim=-1)
        #     out = out[...,:cfg.MODEL.OUTPUT_DIM]
        #     loss = criterion(out*mask.to(device), mask.to(device)*label.to(device).float())+criterion_t(out_t*mask.to(device), mask.to(device)*label_t.to(device).float())
        #     avg_loss += loss.item()
        #     if counter%1 == 0:
        #         print("Epoch {}......Step: {}/{}....... Average Validation Loss for Epoch: {}".format(epoch, counter, len(val_loader), avg_loss/counter))
        avg_loss = train_aux(epoch=epoch, model=model, criterion=criterion, criterion_t=criterion_t, optimizer=optimizer, loader=val_loader, is_training=False, device=device, cfg=cfg)
        val_loss = avg_loss/len(val_loader)
        print("\t\t\tEpoch {}/{} Done, Total validation Loss: {}".format(epoch, cfg.TRAIN.EPOCHS, val_loss))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print("\t\t\tFound new best validation loss (saving model)")
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT.LOCATION, 'step_gru_best_model.pt'))
    return model


def evaluate(model, data_loader, cfg):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.eval()

    outputs = []
    targets = []
    masks   = []
    video_ids = []
    frames = []

    for _, (action, obj, frame, audio, label, _, mask, frame_idx, id) in tqdm.tqdm(enumerate(data_loader)):
      h = model.init_hidden(action.shape[0])
      out, h = model(action.to(device).float(), h, audio.to(device).float(), obj.to(device).float(), frame.to(device).float())

      video_ids.append([i[0] for i in id])
      frames.append(frame_idx.cpu().detach().numpy())
      outputs.append(out.cpu().detach().numpy())
      targets.append(label.numpy())
      masks.append(mask.numpy())

    video_ids = np.concatenate(video_ids)
    frames  = np.concatenate(frames)
    outputs = np.concatenate(outputs)
    targets = np.concatenate(targets)
    masks   = np.concatenate(masks)

    np.save(f'{cfg.OUTPUT.LOCATION}/video_ids.npy', video_ids)
    np.save(f'{cfg.OUTPUT.LOCATION}/frames.npy', frames)
    np.save(f'{cfg.OUTPUT.LOCATION}/outputs.npy', outputs)
    np.save(f'{cfg.OUTPUT.LOCATION}/targets.npy', targets)
    np.save(f'{cfg.OUTPUT.LOCATION}/masks.npy', masks)
