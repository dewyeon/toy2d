from __future__ import print_function
import argparse

__all__ = ['get_args']


def get_args():
    parser = argparse.ArgumentParser(description='CFLOW-AD')
    parser.add_argument('--dataset', default='mvtec', type=str, metavar='D',
                        help='dataset name: mvtec/stc/multi_mvtec (default: mvtec)')
    parser.add_argument('-cl_indexes', '--class_indexes', default='4,5', type=str, metavar='C',
						help='class indexes for multi_MVTec (0~14) (default: 4,5)')
    parser.add_argument('-user', '--user', default='kakao', type=str, help='username: sojin/juyeon/chi/kakao (default: kakao)')
    parser.add_argument('-use_wandb', '--use_wandb', default=True, type=bool)
    parser.add_argument('-wdb_pj', '--wandb_project', default='baselines', type=str)
    parser.add_argument('-wdb', '--wandb_table', default='cflow', type=str)

    parser.add_argument('--checkpoint', default='', type=str, metavar='D',
                        help='file with saved checkpoint')
    parser.add_argument('-cl', '--class-name', default='none', type=str, metavar='C',
                        help='class name for MVTec/STC (default: none)')
    parser.add_argument('-enc', '--enc-arch', default='vit_base_patch16_224', type=str, metavar='A',
                        help='feature extractor: wide_resnet50_2/resnet18/mobilenet_v3_large (default: vit_base_patch16_224)')
    parser.add_argument('-dec', '--dec-arch', default='freia-cflow', type=str, metavar='A',
                        help='normalizing flow model (default: freia-cflow)')
    parser.add_argument('-pl', '--pool-layers', default=3, type=int, metavar='L',
                        help='number of layers used in NF model (default: 3)')
    parser.add_argument('-cb', '--coupling-blocks', default=8, type=int, metavar='L',
                        help='number of layers used in NF model (default: 8)')
    parser.add_argument('-run', '--run-name', default=0, type=int, metavar='C',
                        help='name of the run (default: 0)')
    parser.add_argument('-inp', '--input-size', default=256, type=int, metavar='C',
                        help='image resize dimensions (default: 256)')
    parser.add_argument("--action-type", default='norm-train', type=str, metavar='T',
                        help='norm-train/norm-test (default: norm-train)')
    parser.add_argument('-bs', '--batch-size', default=32, type=int, metavar='B',
                        help='train batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR',
                        help='learning rate (default: 2e-4)')
    parser.add_argument('--meta-epochs', type=int, default=25, metavar='N',
                        help='number of meta epochs to train (default: 25)')
    parser.add_argument('--sub-epochs', type=int, default=8, metavar='N',
                        help='number of sub epochs to train (default: 8)')
    parser.add_argument('--pro', action='store_true', default=False,
                        help='enables estimation of AUPRO metric')
    parser.add_argument('--viz', action='store_true', default=False,
                        help='saves test data visualizations')
    parser.add_argument('--workers', default=4, type=int, metavar='G',
                        help='number of data loading workers (default: 4)')
    parser.add_argument("--gpu", default='0', type=str, metavar='G',
                        help='GPU device number')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--video-path', default='.', type=str, metavar='D',
                        help='video file path')

    args = parser.parse_args()
    
    return args
