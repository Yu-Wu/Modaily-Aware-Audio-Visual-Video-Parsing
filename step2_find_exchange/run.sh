#CUDA_VISIBLE_DEVICES=1 .
python main_avvp.py --mode estimate_labels --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18 --model_save_dir ../step1_train_base_model/models/
