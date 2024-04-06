
GymGuard Client and Trainer:

For running novel training on videos:
1. Put all training videos inside /videos/{label}/**.mp4
2. run `python preprocessor_videos.py`
3. A /trainable_data will be populated with csv files based on the /videos data
4. run `python trainer.py` to train on the /trainable_data files
5. The trainer.py will populate /temp directory with keras model, label mapping text file and other files (loss csv and plots)