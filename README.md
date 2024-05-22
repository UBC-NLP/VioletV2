
# VioletV2
Techincal report coming soon!

## Data
Coco Images HDF5 file: [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_mohamed_mbzuai_ac_ae/EZUDaVbRzGFJuYbYnU_jZ0YBnjZgPSuG32Z6wlLeCT22iQ?e=fI1rG0)

Annotations: [Download](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/abdelrahman_mohamed_mbzuai_ac_ae/EXkwLG9hEE5EimCVLsVpHTwB6EadaXDCXBII3lBquptmjw?e=fD24oa)


## Environment setup
Clone the repository and create the `Violet` conda environmnet


```
conda env create -f violet.yml
```
make logs and saved_models directories

```
mkdir logs
mkdir saved_models
```
## Train the model (refactored code)
### simpler and more friendly impelementation (You can ignore the data folder when using this)
```
python train_refactored.py --batch_size 60 --head 12 --tau 0.3 --images_path coco_images.h5  --annotation_folder annotations --lr 1e-4 --random_seed 42 --log_file logs/log --decoder_layer 12 --optimizer_type adamw  --gradient_accumulation_steps 1  --exp_name violet
```



## Train the model (legacy code)
### based on the code used in Meshed transformer and VisualGPT, edited to use python 3 instead of the original 2.7
```
python train_legacy.py --batch_size 40 --head 12 --tau 0.3 --images_path ./coco_images.h5  --annotation_folder annotations --lr 1e-4 --random_seed 42 --log_file logs/log --decoder_layer 12 --optimizer_type adamw  --gradient_accumulation_steps 1  --exp_name violet
```


## Acknowledgement
This code used resources from [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer), [Transformers](https://github.com/huggingface/transformers) and [VisualGPT](https://github.com/Vision-CAIR/VisualGPT)


