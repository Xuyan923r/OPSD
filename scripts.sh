conda activate /idfsdata/yexuyan/conda_envs/verl

cd /idfsdata/yexuyan/OPSD
mkdir -p logs
rm -f /idfsdata/yexuyan/OPSD/logs/opsd_ifeval_gpu0123.nohup.log

nohup bash -lc '
export PATH=/idfsdata/yexuyan/conda_envs/verl/bin:$PATH
export CUDA_VISIBLE_DEVICES=0,1,2,3
export HF_HOME=/tmp/hf_home_opsd
export HF_DATASETS_CACHE=/tmp/hf_datasets_cache_opsd
export TRITON_CACHE_DIR=/tmp/triton_cache_opsd
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset TRANSFORMERS_CACHE
bash /idfsdata/yexuyan/OPSD/scripts/run_opsd_ifeval.sh
' > /idfsdata/yexuyan/OPSD/logs/opsd_ifeval_gpu0123.nohup.log 2>&1 & echo $!
