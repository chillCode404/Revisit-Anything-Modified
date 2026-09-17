python place_rec_DINO_finetuned.py --dataset pitts --method DINO
# python place_rec_SAM_DINO_finetuned.py --dataset pitts --method SAM
python vlad_c_centers_pt_gen_finetuned.py --dataset pitts --vocab-vlad domain
python place_rec_pca_finetuned.py --dataset pitts --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain
python place_rec_main_finetuned.py --dataset pitts --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain --save-result