export MESA_GL_VERSION_OVERRIDE=4.6
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# conda init
# conda activate env_isaaclab

# export CUDA_VISIBLE_DEVICES=1;python scripts/rsl_rl/train.py --task=Isaac-Velocity-Flat-Unitree-Go2-v0 --num_env 4096 --headless --experiment_name flat_vel --max_iterations 100000 --video

export CUDA_VISIBLE_DEVICES=1;python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint logs/rsl_rl/unitree_go2_flat/2025-12-10_16-13-19/model_22850.pt --enable_cameras --video --num_env 512 --video_length 2000 --headless
# export CUDA_VISIBLE_DEVICES=6;python scripts/rsl_rl/play.py --task=Isaac-Velocity-Flat-Unitree-Go2-v0 --checkpoint logs/rsl_rl/unitree_go2_flat/2025-11-19_19-44-38/model_9999.pt --enable_cameras --video --num_env 512 --video_length 2000 --headless









#export CUDA_VISIBLE_DEVICES=0,1;python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 scripts/rsl_rl/train.py --task=Isaac-Location-Rough-Unitree-Go2-v0 --max_iterations 20000 --num_envs 4096 --video --video_interval 10000 --headless --distributed
# export CUDA_VISIBLE_DEVICES=1;python scripts/rsl_rl/train.py --task=Isaac-Location-Rough-Unitree-Go2-v0 --max_iterations 10000 --num_envs 4096 --video --video_interval 10000 --headless
# export CUDA_VISIBLE_DEVICES=6;python scripts/rsl_rl/play.py --task=Isaac-Location-Rough-Unitree-Go2-v0 --checkpoint logs/rsl_rl/rough_gaits/2025-07-20_22-04-39/model_10000.pt --enable_cameras --video --num_env 512 --video_length 2000 --headless
# export CUDA_VISIBLE_DEVICES=2;python scripts/rsl_rl/train.py --task=Isaac-Location-Flat-Unitree-Go2-v0 --num_env 16384 --headless --experiment_name flat_gaits --max_iterations 100000

# export CUDA_VISIBLE_DEVICES=1;python scripts/rsl_rl/train.py --task=Isaac-Location-Rough-Unitree-Go2-v0 --max_iterations 100000 --num_envs 8192 --headless

# export CUDA_VISIBLE_DEVICES=2;python scripts/rsl_rl/train.py --task=Isaac-Location-Flat-Unitree-Go2-v0  --resume --experiment_name flat_gaits_no_curr --load_run 2025-10-21_10-13-47 --checkpoint model_15000.pt --headless --num_env 16384 --max_iterations 100000


# export CUDA_VISIBLE_DEVICES=1;python scripts/rsl_rl/train.py --task=Isaac-Location-Flat-Unitree-Go2-v0 --num_env 4096 --headless --experiment_name flat_gaits_curr --max_iterations 1000000
# export CUDA_VISIBLE_DEVICES=1,2,3; python -m torch.distributed.run --nnodes=1 --nproc_per_node=3 scripts/rsl_rl/train.py --task=Isaac-Location-Flat-Unitree-Go2-v0 --num_env 8192 --headless --experiment_name flat_gaits_curr_muti_gpu --max_iterations 1000000 --distributed