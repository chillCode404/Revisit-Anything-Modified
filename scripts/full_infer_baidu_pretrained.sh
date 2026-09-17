# python place_rec_SAM_DINO.py --dataset baidu_test --method DINO
# # python place_rec_SAM_DINO.py --dataset baidu_test --method SAM
# python vlad_c_centers_pt_gen.py --dataset baidu_test --vocab-vlad domain
python place_rec_pca.py --dataset baidu_test --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain
python place_rec_main.py --dataset baidu_test --experiment exp0_global_SegLoc_VLAD_PCA_o3 --vocab-vlad domain --save-result