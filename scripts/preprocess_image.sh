source ../envs/diff-spatial/bin/activate
export CUDA_VISIBLE_DEVICES=1

# python preprocess_data.py --dir "./data_original/DIV2K_valid_HR" --outdir "./data_preprocessed/DIV2K_valid_HR"
# python preprocess_data.py --dir "./data_original/Flickr2K" --outdir "./data_preprocessed/Flickr2K"
python preprocess_data.py --dir "./data_original/Flickr1024/Validation" --outdir "./data_preprocessed/Flickr1024_val" --glob_pattern "*_L*"
# python preprocess_data.py --dir "./data_original/HR-WSI/val/imgs" --outdir "./data_preprocessed/HRWSI_val"