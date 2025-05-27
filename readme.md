## ConvNext with MoE block



### install
`
conda create -n convnext_moe python=3.10
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
pip install numpy==1.26

pip install -U openmim
mim install mmengine
mim install mmcv==2.1

cd mmdetection
pip install -v -e .

pip install git+https://github.com/JonathonLuiten/TrackEval.git
`