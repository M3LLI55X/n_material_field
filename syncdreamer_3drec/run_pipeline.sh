module load sqlite3/3.42.0 tensorrt/8.6.1.6-cuda-12.X cmake/3.28.3 openblas/0.3.23 cudnn/v8.9.7.29-prod-cuda-12.X cuda/12.1 gcc/11.5.0-binutils-2.43 blender/3.6.2

source /dtu/blackhole/11/180913/Miniconda3/bin/activate

conda activate zero123

cd /dtu/blackhole/11/180913/n_material_field/syncdreamer_3drec

python -m ipdb my_train_renderer_spin36.py -i /dtu/blackhole/11/180913/n_material_field/syncdreamer_3drec/chair_5_rgba -n chair_5 -e 70 -d 1.5 -l test_for_seg0