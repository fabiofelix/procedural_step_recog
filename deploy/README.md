# PTG Setup



## System Setup
1. Install docker
2. Install docker compose (`pip install docker-compose`)
3. Install [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Install
### Clone repo
```bash
# clone
git clone https://github.com/fabiofelix/procedural_step_recog.git
cd procedural_step_recog/deploy
```
### Download models
You can put `/home/ptg/data` wherever, just be consistent.
```bash
# setup data dir
mkdir -p /home/ptg/data
mkdir -p /home/ptg/data/models
cp -r ~/Downloads/models /home/ptg/data/models  # change this to pull from google drive or something
```
### Create .env file
```bash
VOLUMES=/home/ptg/data
```
### Deploy API
```bash
# Deploy the API server
docker compose -f docker-compose.api.yaml up -d --build
```
### Deploy models
```bash
# Deploy machine learning model
docker compose up -d --build
```
## Install ZMQ Adapters
### Clone repo
```bash
git clone https://github.com/VIDA-NYU/bbn-comms.git
cd bbn-comms
```
### Create .env file
Replace the IPs with the IP of the machine you want to talk to:
```bash
VOLUMES=/home/ptg/data
REDIS_URL=redis://128.1.1.172:6379
BBN_CTL_URL=tcp://128.1.1.172:5555
BBN_MSG_URL=tcp://128.1.1.172:6666
BBN_MSG_URL2=tcp://128.1.1.172:6670
```
### Deploy
```bash
docker compose up -d --build
```
