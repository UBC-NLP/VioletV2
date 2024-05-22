
# VioletV2
Techincal report coming soon!


## Environment setup
Clone the repository and create the `Violet` conda environmnet


```
conda env create -f violet.yml
```

## Train the model (refactored code)
```
python train_refactored.py --batch_size 40 --head 12 --tau 0.3 --images_path coco_images.h5  --annotation_folder annotations --lr 1e-4 --random_seed 42 --log_file logs/log --decoder_layer 12 --optimizer_type adamw  --gradient_accumulation_steps 1  --exp_name violet
```



## Train the model (legacy code)
```
python train_legacy.py --batch_size 40 --head 12 --tau 0.3 --images_path ./coco_images.h5  --annotation_folder annotations --lr 1e-4 --random_seed 42 --log_file logs/log --decoder_layer 12 --optimizer_type adamw  --gradient_accumulation_steps 1  --exp_name violet
```

## Train the model (refactored code)
```
python train_refactored.py --batch_size 40 --head 12 --tau 0.3 --images_path ./coco_images.h5  --annotation_folder annotations --lr 1e-4 --random_seed 42 --log_file logs/log --decoder_layer 12 --optimizer_type adamw  --gradient_accumulation_steps 1  --exp_name violet"
```

## Acknowledgement
This code used resources from [Meshed Memory Transformer](https://github.com/aimagelab/meshed-memory-transformer), [Transformers](https://github.com/huggingface/transformers) and [VisualGPT](https://github.com/Vision-CAIR/VisualGPT)


