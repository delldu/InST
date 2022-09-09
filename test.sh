# python test_LGW.py \
# 	--source_path images/running-images-mask/source_rgb/juice.png \
# 	--source_mask_path images/running-images-mask/source/juice.png \
# 	--target_mask_path images/running-images-mask/target/vase5_mask.jpg \
# 	--checkpoint pretrained/checkpoint_r3.pth

python test_LGW.py \
	--source_dir images/running-images-mask/source_rgb \
	--source_mask_dir images/running-images-mask/source \
	--target_mask_dir images/running-images-mask/target \
	--checkpoint pretrained/checkpoint_r3.pth
