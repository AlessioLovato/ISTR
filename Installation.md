# Installing ISTR
This readme explains the procedure on how to install and run ISTR github repo.<br>
The original repo is available at https://github.com/hujiecpp/ISTR/tree/master.

## Installation
1. Install pip venv<br>
Normal installation
```bash
sudo apt-get update && apt-get install -y python3 python3-venv python3-pip
python3 -m venv venv
```
Dockerfile installation (automatically activated)
```
RUN apt-get update && apt-get install -y python3 python3-venv python3-pip
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
```

2. Install packages in venv
```bash
sudo apt install -y libgl1
source venv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip3 install opencv-python scipy shapely torch_dct timm
```

3. Download the github repo
```bash
git clone https://github.com/AlessioLovato/ISTR.git
```

4. Install the repo
```bash
cd ISTR
pip install -e detectron2 --no-build-isolation
pip install -e . --no-build-isolation
```