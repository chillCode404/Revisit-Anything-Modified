python place_rec_DINO_finetuned.py --dataset baidu_test
# python place_rec_SAM_DINO.py --dataset baidu_test --method SAM
python vlad_c_centers_pt_gen_finetuned.py --dataset baidu_test --vocab-vlad domain
python place_rec_pca_finetuned.py --dataset baidu_test --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain
python place_rec_main_finetuned.py --dataset baidu_test --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain --save-result