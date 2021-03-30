# Exploring Heterogeneous Clues for Weakly Supervised Audio-Visual Video Parsing
Code for CVPR 2021 paper [_Exploring Heterogeneous Clues for Weakly-Supervised Audio-Visual Video Parsing_](https://yu-wu.net/pdf/CVPR21_audio.pdf)


## The Audio-Visual Video Parsing task
We aim at identifying the audible and visible events and their temporal location in videos. Note that the visual and audio events might be asynchronous.
<div align=center><img src="https://github.com/Yu-Wu/Modaily-Aware-Audio-Visual-Video-Parsing/blob/master/task.png" width="600"></div>


## Prepare data
Please refer to https://github.com/YapengTian/AVVP-ECCV20 for downloading the LLP Dataset and the preprocessed audio and visual features.
Put the downloaded `r2plus1d_18`, `res152`, `vggish` features into the `feats` folder.


## Training pipeline
The training includes three stages.

### Train a base model
We first train a base model using MIL and our proposed contrastive learning.
```shell
cd step1_train_base_model
python main_avvp.py --mode train --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18
```


### Generate modality-aware labels
We then freeze the trained model and evaluate each video by swapping its audio and visual tracks with other unrelated videos.
```shell
cd step2_find_exchange
python main_avvp.py --mode estimate_labels --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18
```

### Re-train using modality-aware labels
We then re-train the model from scratch using modality-aware labels.
```shell
cd step3_retrain
python main_avvp.py --mode retrain --audio_dir ../feats/vggish/ --video_dir ../feats/res152/ --st_dir ../feats/r2plus1d_18
```


If you are interested in our paper and want to access the training code, please feel free to contact us by email yu.wu-3@student.uts.edu.au



## Citation

Please cite the following paper in your publications if it helps your research:


    @inproceedings{wu2021explore,
        title = {Exploring Heterogeneous Clues for Weakly-Supervised Audio-Visual Video Parsing},
        author = {Wu, Yu and Yang, Yi},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2021}
        
    }
    
