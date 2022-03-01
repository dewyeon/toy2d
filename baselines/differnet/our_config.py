from __future__ import print_function
import argparse

__all__ = ['get_args']

MVTEC_CLASS_NAMES = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
               'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
               'tile', 'toothbrush', 'transistor', 'wood', 'zipper']
# conda activate anomaly; cd anomaly/ICML2022/multi-modal_AD
# conda activate anomaly;  cd anomaly/multi-modal_AD
# python main.py --meta_epochs 25 --sub_epochs 20 --lr 1e-4 --step_size 10 --lr_warm_epochs 5 --warm_up_adj 10 --gpu 0 -wdb 'warm_up adj'
# --lr, --step_size, --lr_warm_epochs, --warm_up_adj
# conv_attn: 16, 32, 64, 196, 256
# condition_vec : 128, 64, 32, 8

# c.lr_cosine for only adam
##################33done abov0
# to do conv_attn 16,32

##########   default warm_up_epoch 2
# 4,8,16,32,64,128,256,512,768,1024,2048,4096
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K1' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 1 -bv 'Base+our' -tv '0-1' --gpu 1
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K2' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 2 -bv 'Base+our' -tv '0-1' --gpu 2
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K3' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 3 -bv 'Base+our' -tv '0-1' --gpu 3
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K4' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 4 -bv 'Base+our' -tv '0-1' --gpu 4
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K5' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 5 -bv 'Base+our' -tv '0-1' --gpu 5
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K6' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 6 -bv 'Base+our' -tv '0-1' --gpu 6
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K7' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 7 -bv 'Base+our' -tv '0-1' --gpu 7
# python main.py --meta_epochs 25 --sub_epochs 20 -wdb 'Base+Our_K8' --lr 1e-4 --conv_attn 128 --condition_vec 128 -K 8 -bv 'Base+our' -tv '0-1' --gpu 0




# 

# cable, capsule, hazelnut, pill, toothbrush, transistor
# hazelnut_cable, zipper_toothbrush
def get_args():
    parser = argparse.ArgumentParser(description='CS-FLOW')

    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name mvtec/stc/multi_mvtec/toy_example (default: mvtec)')
    parser.add_argument('-cl_indexes', '--class_indexes', default='0,1,2,3,4,5,6,7,8,9,10,11,12,13,14', type=str, metavar='C', help='class indexes for multi_MVTec (0~14) (default: 4,5)')
    parser.add_argument('-user', '--user', default='sojin', type=str, help='username: sojin/juyeon/chi/kakao (default: mvtec)')
  
    parser.add_argument('-use_wandb', '--use_wandb', default=True, type=bool)
    parser.add_argument('-wdb_pj', '--wandb_project', default='baselines', type=str)
    parser.add_argument('-wdb', '--wandb_table', default='differNet', type=str)
    
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--video-path', default='.', type=str, metavar='D',
                        help='video file path')

    args = parser.parse_args()
    
    return args

# python main.py --action-type toy-example --dataset toy_example 
# decoder type-freia-cflow & is_attention False : original cflow-ad
# decoder type-freia-cflow & is_attention True : original cflow-ad + isdp attention
