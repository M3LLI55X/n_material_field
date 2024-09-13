# Modify from ViewFusion
###  Novel View Synthesis
same environment with [Zero-1-to-3](https://github.com/cvlab-columbia/zero123).
```
conda create -n zero123 python=3.9
conda activate zero123
cd zero123
pip install -r requirements.txt
git clone https://github.com/CompVis/taming-transformers.git
pip install -e taming-transformers/
git clone https://github.com/openai/CLIP.git
pip install -e CLIP/
```

Download checkpoint under `zero123` through one of the following sources:

```
https://huggingface.co/cvlab/zero123-weights/tree/main
wget https://cv.cs.columbia.edu/zero123/assets/105000.ckpt    # iteration = [105000, 165000, 230000, 300000]
wget https://zero123.cs.columbia.edu/assets/zero123-xl.ckpt
```
[Zero-1-to-3](https://github.com/cvlab-columbia/zero123) has released 5 model weights: `105000.ckpt`, `165000.ckpt`, `230000.ckpt`, `300000.ckpt`, and `zero123-xl.ckpt`. By default, we use `zero123-xl.ckpt`, but we also find that 105000.ckpt which is the checkpoint after finetuning 105000 iterations on objaverse has better generalization ablitty. So if you are trying to generate novel-view images and find one model fails, you can try another one.

We have provided some processed real images in `./3drec/data/real_images/`. You can directly run `generate_360_view_autoregressive.py` and play it. Don't forget to change the model path and configure path in the code.
```
cd ./3drec/data/
python generate_360_view_autoregressive_real.py # generate 360-degree images for real image demo
python generate_360_view_autoregressive.py # generate 360-degree images for multi-view consistency evaluation
python generate_zero123renderings_autoregressive.py # generate only 1 novel-view image given target pose for image quality evaluation
```

If you want to try it on your own images, you may need to pre-process them, including resize and segmentation. 
```
cd ./3drec/data/
python process_real_images.py
```
If you find any other interesting images that can be shown here, please send it to me and I'm very happy to make our project more attractive! :wink:


### 3D Reconstruction (NeuS)
Note that we haven't use the distillation way to get the 3D model, no matter [SDS](https://github.com/ashawkey/stable-dreamfusion) or [SJC](https://github.com/pals-ttic/sjc). We directly train [NeuS](https://github.com/Totoro97/NeuS) as our model can generate consistent multi-view images. Feel free to explore and play around!
```
cd 3drec
pip install -r requirements.txt
cd ../syncdreamer_3drec

python my_train_renderer_spin36.py -i /dtu/blackhole/11/180913/test/ViewFusion/syncdreamer_3drec/pretrain_zero123_xl_360_autoregressive_real/chair_5_rgba -n chair_5 -e 70 -d 1.5 -l test_imgnew
```
- You can see results under: `syncdreamer_3drec/{output_dir}/{model_name}`.  If you are trying on real images and have no idea about the `evaluation` and `distance`, maybe you can set them as default `60/180*pi` and `1.5`, respectively.



##  Work on DTU HPC

```
module load sqlite3/3.42.0 tensorrt/8.6.1.6-cuda-12.X cmake/3.28.3  openblas/0.3.23 cudnn/v8.9.7.29-prod-cuda-12.X cuda/12.1 gcc/11.5.0-binutils-2.43    blender/3.6.2
```
multi-view generation
```
python /dtu/blackhole/11/180913/test/ViewFusion/zero123/generate_360_view_autoregressive_real.py
```

# conda list
```
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                 conda_forge    conda-forge
_openmp_mutex             4.5                       2_gnu    conda-forge
absl-py                   2.1.0                    pypi_0    pypi
accelerate                0.23.0                   pypi_0    pypi
aiofiles                  23.2.1                   pypi_0    pypi
aiohappyeyeballs          2.4.0                    pypi_0    pypi
aiohttp                   3.10.5                   pypi_0    pypi
aiosignal                 1.3.1                    pypi_0    pypi
albumentations            0.4.3                    pypi_0    pypi
aliyun-python-sdk-core    2.15.2                   pypi_0    pypi
aliyun-python-sdk-kms     2.16.5                   pypi_0    pypi
altair                    5.4.1                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
antlr4-python3-runtime    4.8                      pypi_0    pypi
anyio                     4.4.0                    pypi_0    pypi
appdirs                   1.4.4                    pypi_0    pypi
apptools                  5.3.0                    pypi_0    pypi
asttokens                 2.4.1              pyhd8ed1ab_0    conda-forge
astunparse                1.6.3                    pypi_0    pypi
async-timeout             4.0.3                    pypi_0    pypi
attrs                     24.2.0                   pypi_0    pypi
blinker                   1.8.2                    pypi_0    pypi
braceexpand               0.1.7                    pypi_0    pypi
ca-certificates           2024.8.30            hbcca054_0    conda-forge
cachetools                5.5.0                    pypi_0    pypi
carvekit-colab            4.1.0                    pypi_0    pypi
certifi                   2024.7.4                 pypi_0    pypi
cffi                      1.17.1                   pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
cityscapesscripts         2.2.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
clip                      1.0                       dev_0    <develop>
coloredlogs               15.0.1                   pypi_0    pypi
comm                      0.2.2              pyhd8ed1ab_0    conda-forge
configobj                 5.0.8                    pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
crcmod                    1.7                      pypi_0    pypi
cryptography              43.0.1                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
cython                    3.0.2                    pypi_0    pypi
datasets                  2.4.0                    pypi_0    pypi
debugpy                   1.8.5            py39hf88036b_1    conda-forge
decorator                 5.1.1              pyhd8ed1ab_0    conda-forge
deepspeed                 0.10.3                   pypi_0    pypi
diffdist                  0.1                      pypi_0    pypi
diffusers                 0.12.1                   pypi_0    pypi
dill                      0.3.5.1                  pypi_0    pypi
docker-pycreds            0.4.0                    pypi_0    pypi
easydict                  1.13                     pypi_0    pypi
einops                    0.7.0                    pypi_0    pypi
envisage                  7.0.3                    pypi_0    pypi
exceptiongroup            1.2.2              pyhd8ed1ab_0    conda-forge
executing                 2.0.1                    pypi_0    pypi
fastapi                   0.112.2                  pypi_0    pypi
fastcore                  1.7.1                    pypi_0    pypi
fastjsonschema            2.20.0                   pypi_0    pypi
ffmpeg-python             0.2.0                    pypi_0    pypi
ffmpy                     0.4.0                    pypi_0    pypi
filelock                  3.14.0                   pypi_0    pypi
fire                      0.4.0                    pypi_0    pypi
flask                     2.0.3                    pypi_0    pypi
flatbuffers               24.3.25                  pypi_0    pypi
fonttools                 4.53.1                   pypi_0    pypi
frozenlist                1.4.1                    pypi_0    pypi
fsspec                    2024.6.1                 pypi_0    pypi
ftfy                      6.1.1                    pypi_0    pypi
future                    1.0.0                    pypi_0    pypi
fvcore                    0.1.5.post20221221          pypi_0    pypi
gast                      0.6.0                    pypi_0    pypi
gitdb                     4.0.11                   pypi_0    pypi
gitpython                 3.1.43                   pypi_0    pypi
google-pasta              0.2.0                    pypi_0    pypi
gradio                    3.42.0                   pypi_0    pypi
gradio-client             0.5.0                    pypi_0    pypi
grpcio                    1.66.0                   pypi_0    pypi
h11                       0.14.0                   pypi_0    pypi
h5py                      3.11.0                   pypi_0    pypi
hjson                     3.1.0                    pypi_0    pypi
httpcore                  1.0.5                    pypi_0    pypi
httpx                     0.27.1                   pypi_0    pypi
huggingface-hub           0.17.3                   pypi_0    pypi
humanfriendly             10.0                     pypi_0    pypi
idna                      3.8                      pypi_0    pypi
imageio                   2.35.1                   pypi_0    pypi
imageio-ffmpeg            0.4.2                    pypi_0    pypi
imgaug                    0.2.6                    pypi_0    pypi
importlib-metadata        8.4.0              pyha770c72_0    conda-forge
importlib-resources       6.4.4                    pypi_0    pypi
importlib_metadata        8.4.0                hd8ed1ab_0    conda-forge
infinibatch               0.1.1                    pypi_0    pypi
iopath                    0.1.10                   pypi_0    pypi
ipykernel                 6.29.5             pyh3099207_0    conda-forge
ipython                   8.18.1             pyh707e725_3    conda-forge
itsdangerous              2.2.0                    pypi_0    pypi
jedi                      0.19.1             pyhd8ed1ab_0    conda-forge
jinja2                    3.1.4                    pypi_0    pypi
jmespath                  0.10.0                   pypi_0    pypi
joblib                    1.4.2                    pypi_0    pypi
json-tricks               3.17.3                   pypi_0    pypi
jsonschema                4.23.0                   pypi_0    pypi
jsonschema-specifications 2023.12.1                pypi_0    pypi
jupyter_client            8.6.2              pyhd8ed1ab_0    conda-forge
jupyter_core              5.7.2            py39hf3d152e_0    conda-forge
kaolin                    0.13.0                   pypi_0    pypi
keras                     3.5.0                    pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
kornia                    0.6.8                    pypi_0    pypi
lazy-loader               0.4                      pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libclang                  18.1.1                   pypi_0    pypi
libffi                    3.4.4                h6a678d5_1  
libgcc                    14.1.0               h77fa898_1    conda-forge
libgcc-ng                 14.1.0               h69a702a_1    conda-forge
libgomp                   14.1.0               h77fa898_1    conda-forge
libsodium                 1.0.18               h36c2ea0_1    conda-forge
libstdcxx                 14.1.0               hc0a3c3a_1    conda-forge
libstdcxx-ng              11.2.0               h1234567_1  
linkify-it-py             2.0.3                    pypi_0    pypi
llvmlite                  0.43.0                   pypi_0    pypi
loguru                    0.7.2                    pypi_0    pypi
lovely-numpy              0.2.13                   pypi_0    pypi
lovely-tensors            0.1.16                   pypi_0    pypi
lpips                     0.1.4                    pypi_0    pypi
markdown                  3.7                      pypi_0    pypi
markdown-it-py            2.2.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.2                    pypi_0    pypi
matplotlib-inline         0.1.7              pyhd8ed1ab_0    conda-forge
mayavi                    4.8.2                    pypi_0    pypi
mdit-py-plugins           0.3.3                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
ml-dtypes                 0.4.0                    pypi_0    pypi
more-itertools            10.5.0                   pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
multidict                 6.0.5                    pypi_0    pypi
multiprocess              0.70.13                  pypi_0    pypi
mup                       1.0.0                    pypi_0    pypi
namex                     0.0.8                    pypi_0    pypi
narwhals                  1.5.5                    pypi_0    pypi
nbformat                  5.10.4                   pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
nest-asyncio              1.6.0              pyhd8ed1ab_0    conda-forge
networkx                  3.2.1                    pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
nltk                      3.8.1                    pypi_0    pypi
numba                     0.60.0                   pypi_0    pypi
numpy                     1.23.1                   pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.6.20                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
omegaconf                 2.1.1                    pypi_0    pypi
onnxruntime               1.19.0                   pypi_0    pypi
openai-whisper            20230306                 pypi_0    pypi
opencv-python             4.8.1.78                 pypi_0    pypi
opencv-python-headless    4.10.0.84                pypi_0    pypi
openssl                   3.3.2                hb9d3cd8_0    conda-forge
openxlab                  0.1.1                    pypi_0    pypi
opt-einsum                3.3.0                    pypi_0    pypi
optree                    0.12.1                   pypi_0    pypi
orjson                    3.10.7                   pypi_0    pypi
oss2                      2.17.0                   pypi_0    pypi
packaging                 24.1               pyhd8ed1ab_0    conda-forge
pandas                    2.0.3                    pypi_0    pypi
parso                     0.8.4              pyhd8ed1ab_0    conda-forge
pathtools                 0.1.2                    pypi_0    pypi
pexpect                   4.9.0              pyhd8ed1ab_0    conda-forge
pickleshare               0.7.5                   py_1003    conda-forge
pillow                    9.4.0                    pypi_0    pypi
pip                       24.2             py39h06a4308_0  
platformdirs              4.2.2              pyhd8ed1ab_0    conda-forge
plotly                    5.13.1                   pypi_0    pypi
point-cloud-utils         0.30.4                   pypi_0    pypi
pooch                     1.8.2                    pypi_0    pypi
portalocker               2.10.1                   pypi_0    pypi
prompt-toolkit            3.0.47             pyha770c72_0    conda-forge
protobuf                  4.25.4                   pypi_0    pypi
psutil                    6.0.0            py39hd3abc70_0    conda-forge
ptyprocess                0.7.0              pyhd3deb0d_0    conda-forge
pudb                      2019.2                   pypi_0    pypi
pure_eval                 0.2.3              pyhd8ed1ab_0    conda-forge
py-cpuinfo                9.0.0                    pypi_0    pypi
pyarrow                   13.0.0                   pypi_0    pypi
pycocotools               2.0.7                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pycryptodome              3.20.0                   pypi_0    pypi
pydantic                  1.10.18                  pypi_0    pypi
pydantic-core             2.20.1                   pypi_0    pypi
pydeck                    0.9.1                    pypi_0    pypi
pydeprecate               0.3.1                    pypi_0    pypi
pydub                     0.25.1                   pypi_0    pypi
pyface                    8.0.0                    pypi_0    pypi
pygments                  2.18.0             pyhd8ed1ab_0    conda-forge
pymatting                 1.1.12                   pypi_0    pypi
pymcubes                  0.1.6                    pypi_0    pypi
pyparsing                 3.1.4                    pypi_0    pypi
pyqt5                     5.15.11                  pypi_0    pypi
pyqt5-qt5                 5.15.14                  pypi_0    pypi
pyqt5-sip                 12.15.0                  pypi_0    pypi
pyquaternion              0.9.9                    pypi_0    pypi
python                    3.9.19               h955ad1f_1  
python-dateutil           2.9.0.post0              pypi_0    pypi
python-multipart          0.0.9                    pypi_0    pypi
python_abi                3.9                      2_cp39    conda-forge
pytorch-lightning         1.4.2                    pypi_0    pypi
pytz                      2023.4                   pypi_0    pypi
pywavelets                1.6.0                    pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
pyzmq                     26.2.0           py39h4e4fb57_1    conda-forge
readline                  8.2                  h5eee18b_0  
referencing               0.35.1                   pypi_0    pypi
regex                     2023.10.3                pypi_0    pypi
rembg                     2.0.59                   pypi_0    pypi
requests                  2.28.2                   pypi_0    pypi
responses                 0.18.0                   pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
rpds-py                   0.20.0                   pypi_0    pypi
safetensors               0.4.5                    pypi_0    pypi
scikit-image              0.21.0                   pypi_0    pypi
scikit-learn              1.3.1                    pypi_0    pypi
scipy                     1.9.1                    pypi_0    pypi
seaborn                   0.13.2                   pypi_0    pypi
segment-anything          1.0                      pypi_0    pypi
semantic-sam              1.0                      pypi_0    pypi
semantic-version          2.10.0                   pypi_0    pypi
sentencepiece             0.1.99                   pypi_0    pypi
sentry-sdk                2.14.0                   pypi_0    pypi
setproctitle              1.3.3                    pypi_0    pypi
setuptools                60.2.0                   pypi_0    pypi
shapely                   1.8.0                    pypi_0    pypi
six                       1.16.0             pyh6c4a22f_0    conda-forge
smmap                     5.0.1                    pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
stack-data                0.6.3                    pypi_0    pypi
stack_data                0.6.2              pyhd8ed1ab_0    conda-forge
starlette                 0.38.2                   pypi_0    pypi
streamlit                 1.37.1                   pypi_0    pypi
sympy                     1.13.2                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
taming-transformers       0.0.1                     dev_0    <develop>
tenacity                  8.5.0                    pypi_0    pypi
tensorboard               2.17.1                   pypi_0    pypi
tensorboard-data-server   0.7.2                    pypi_0    pypi
tensorflow                2.17.0                   pypi_0    pypi
tensorflow-io-gcs-filesystem 0.37.1                   pypi_0    pypi
termcolor                 2.4.0                    pypi_0    pypi
test-tube                 0.7.5                    pypi_0    pypi
threadpoolctl             3.5.0                    pypi_0    pypi
tifffile                  2024.8.24                pypi_0    pypi
timm                      0.4.12                   pypi_0    pypi
tinycudann                1.7                      pypi_0    pypi
tk                        8.6.14               h39e8969_0  
tokenizers                0.14.1                   pypi_0    pypi
toml                      0.10.2                   pypi_0    pypi
torch                     2.4.0                    pypi_0    pypi
torch-fidelity            0.3.0                    pypi_0    pypi
torchaudio                2.4.0                    pypi_0    pypi
torchmetrics              0.6.0                    pypi_0    pypi
torchvision               0.19.0+cu121             pypi_0    pypi
tornado                   6.1                      pypi_0    pypi
tqdm                      4.65.2                   pypi_0    pypi
traitlets                 5.14.3             pyhd8ed1ab_0    conda-forge
traits                    6.4.3                    pypi_0    pypi
traitsui                  8.0.0                    pypi_0    pypi
transformers              4.34.0                   pypi_0    pypi
trimesh                   4.4.7                    pypi_0    pypi
triton                    3.0.0                    pypi_0    pypi
typing                    3.7.4.3                  pypi_0    pypi
typing_extensions         4.12.2             pyha770c72_0    conda-forge
tzdata                    2024.1                   pypi_0    pypi
uc-micro-py               1.0.3                    pypi_0    pypi
urllib3                   1.26.20                  pypi_0    pypi
urwid                     2.6.15                   pypi_0    pypi
usd-core                  22.5.post1               pypi_0    pypi
uvicorn                   0.30.6                   pypi_0    pypi
vision-datasets           0.2.2                    pypi_0    pypi
vtk                       9.3.1                    pypi_0    pypi
wandb                     0.15.12                  pypi_0    pypi
watchdog                  4.0.2                    pypi_0    pypi
wcwidth                   0.2.13             pyhd8ed1ab_0    conda-forge
webdataset                0.2.5                    pypi_0    pypi
websockets                11.0.3                   pypi_0    pypi
werkzeug                  3.0.4                    pypi_0    pypi
wheel                     0.43.0           py39h06a4308_0  
wrapt                     1.16.0                   pypi_0    pypi
xxhash                    3.5.0                    pypi_0    pypi
xz                        5.4.6                h5eee18b_1  
yacs                      0.1.8                    pypi_0    pypi
yarl                      1.9.4                    pypi_0    pypi
zeromq                    4.3.5                h6a678d5_0  
zipp                      3.20.1             pyhd8ed1ab_0    conda-forge
zlib                      1.2.13               h5eee18b_1
```