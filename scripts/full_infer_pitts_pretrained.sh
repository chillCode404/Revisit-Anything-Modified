python place_rec_SAM_DINO.py --dataset pitts --method DINO
# python place_rec_SAM_DINO.py --dataset pitts --method SAM
python vlad_c_centers_pt_gen.py --dataset pitts --vocab-vlad domain
python place_rec_pca.py --dataset pitts --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain
python place_rec_main.py --dataset pitts --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain --save-result