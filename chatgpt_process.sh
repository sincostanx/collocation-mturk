source envs/diff-spatial/bin/activate
export CUDA_VISIBLE_DEVICES=0

python chatgpt_process.py \
    --df_path new_selected_dataset_eyeballing.csv