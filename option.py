import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='TTSR-lowlight — Reference-based Low-Light Enhancement')

### log setting
parser.add_argument('--save_dir', type=str, default='/root/data/experiments/TTSR-lowlight',
                    help='Directory to save log, arguments, models and images')
parser.add_argument('--reset', type=str2bool, default=False,
                    help='Delete save_dir to create a new one')
parser.add_argument('--log_file_name', type=str, default='TTSR-lowlight.log',
                    help='Log file name')
parser.add_argument('--logger_name', type=str, default='TTSR-lowlight',
                    help='Logger name')

### device setting
parser.add_argument('--cpu', type=str2bool, default=False,
                    help='Use CPU to run code')
parser.add_argument('--num_gpu', type=int, default=1,
                    help='The number of GPU used in training')
parser.add_argument('--seed', type=int, default=-1,
                    help='Random seed for torch/numpy/random. -1 disables seeding '
                         '(use a fixed value when comparing ablations)')

### dataset setting
parser.add_argument('--dataset', type=str, default='LOL',
                    help='Which dataset to train and test [CUFED / LOL / data1]')
parser.add_argument('--dataset_dir', type=str, default='/root/data/datasets/LOLdataset',
                    help='Directory of dataset')
parser.add_argument('--data1_camera', type=str, default='all',
                    help='Camera subset for data1 eval: all / Huawei / Nikon')
parser.add_argument('--ref_dir', type=str, default='',
                    help='External reference directory (e.g. nanobanana refs). Overrides HR-as-ref.')


### reference degradation (v2: simulate real-world ref mismatch)
parser.add_argument('--ref_degrade', type=str2bool, default=True,
                    help='Apply color jitter + spatial shift + blur to reference image')
parser.add_argument('--ref_color_jitter', type=float, default=0.2,
                    help='Color jitter strength for ref (brightness/contrast/saturation)')
parser.add_argument('--ref_shift_range', type=int, default=4,
                    help='Max pixel shift for ref spatial misalignment')
parser.add_argument('--ref_blur_sigma', type=float, default=2.0,
                    help='Max Gaussian blur sigma for ref')
parser.add_argument('--ref_gamma', type=float, default=1.0,
                    help='Gamma applied to external nanobanana refs only. 1.0 = no change')
parser.add_argument('--eval_degrade_seed', type=int, default=1234,
                    help='Fixed seed for eval-time ref degradation so GT eval is reproducible')
parser.add_argument('--eval_ref_degrade', type=str2bool, default=None,
                    help='Override ref degradation for the test loader only; None = follow --ref_degrade')
parser.add_argument('--ref_degrade_nanobanana', type=str2bool, default=False,
                    help='Apply mild ref degradation to nanobanana refs during mixed training')

### dataloader setting
parser.add_argument('--num_workers', type=int, default=4,
                    help='The number of workers when loading data')

### mode setting
parser.add_argument('--enhance_mode', type=str2bool, default=True,
                    help='Use low-light enhancement mode (1:1 resolution) instead of super-resolution (4x)')
parser.add_argument('--enhance_backbone', type=str, default='ttsr',
                    choices=['ttsr', 'retinexformer'],
                    help='Enhancement backbone: "ttsr" = MainNetEnhance (default, unchanged), '
                         '"retinexformer" = Retinexformer + H/4 texture adapter')
parser.add_argument('--retinex_n_feat', type=int, default=40,
                    help='Retinexformer base width (official LOL-v1 setting: 40)')
parser.add_argument('--retinex_num_blocks', type=str, default='1,2,2',
                    help='Retinexformer num_blocks, comma separated (official LOL-v1: 1,2,2)')
parser.add_argument('--no_ref_texture', type=str2bool, default=False,
                    help='Disable only the texture path (LTE search/transfer + adapter). '
                         'RefIllumTransfer keeps receiving the reference.')
parser.add_argument('--texture_lv3_only', type=str2bool, default=False,
                    help='CONTROL for the backbone comparison: keep only the T_lv3 injection '
                         '(stage 1) on the ttsr backbone, disabling the T_lv2/T_lv1 '
                         'injections. Lets a backbone swap be separated from a change of the '
                         'texture injection topology. Ignored by the retinexformer backbone, '
                         'which injects T_lv3 only by construction.')
parser.add_argument('--adapter_only', type=str2bool, default=False,
                    help='Freeze the whole network except the H/4 texture adapter (A3) and '
                         'train only those ~0.30M parameters. Uses a step-based budget '
                         '(--adapter_steps) instead of epochs. Requires --adapter_base_ckpt.')
parser.add_argument('--adapter_base_ckpt', type=str, default='',
                    help='Checkpoint loaded as the frozen base for --adapter_only '
                         '(e.g. the full-data N0 ep40). Must exist; no fallback.')
parser.add_argument('--adapter_steps', type=int, default=3000,
                    help='Optimizer updates for --adapter_only')
parser.add_argument('--adapter_lr', type=float, default=1e-4,
                    help='Adapter LR before the drop')
parser.add_argument('--adapter_lr_drop_step', type=int, default=2000,
                    help='LR is halved once this many updates have completed')
parser.add_argument('--adapter_lr_after_drop', type=float, default=5e-5,
                    help='Adapter LR after the drop')
parser.add_argument('--adapter_eval_every', type=int, default=1000,
                    help='Run a full evaluation every N adapter updates (step 0 always runs)')
parser.add_argument('--train_arm', type=str, default='none',
                    choices=['none', 'adapter', 'decoder_tail', 'decoder_tail_texture'],
                    help='Step-budget training arm. "adapter" = only the H/4 texture adapter '
                         '(same as --adapter_only); "decoder_tail" = last decoder stage + '
                         'mapping only (no reference); "decoder_tail_texture" = decoder_tail '
                         'plus the texture adapter. Everything else stays frozen.')
parser.add_argument('--train_base_ckpt', type=str, default='',
                    help='Frozen starting checkpoint for --train_arm (falls back to '
                         '--adapter_base_ckpt). Must exist; no fallback.')
parser.add_argument('--train_steps', type=int, default=3000,
                    help='Optimizer updates for --train_arm')
parser.add_argument('--train_lr_drop_step', type=int, default=2000,
                    help='LRs are reduced once this many updates have completed')
parser.add_argument('--train_eval_every', type=int, default=1000,
                    help='Full evaluation every N updates (step 0 always runs)')
parser.add_argument('--tail_lr', type=float, default=1e-5,
                    help='LR for the decoder-tail group before the drop')
parser.add_argument('--tail_lr_after_drop', type=float, default=5e-6,
                    help='LR for the decoder-tail group after the drop')
parser.add_argument('--no_global_illum', type=str2bool, default=False,
                    help='Disable GlobalIllumHead entirely (not constructed, so it adds no '
                         'parameters and no post-processing step)')

### model setting
parser.add_argument('--num_res_blocks', type=str, default='8+8+4+2',
                    help='The number of residual blocks in each stage (lighter for 1:1 enhancement)')
parser.add_argument('--n_feats', type=int, default=64,
                    help='The number of channels in network')
parser.add_argument('--res_scale', type=float, default=1.,
                    help='Residual scale')
parser.add_argument('--load_pretrain', type=str2bool, default=False,
                    help='Load pre-trained TTSR weights for transfer learning')
parser.add_argument('--pretrain_path', type=str, default='/root/data/pretrain_models/TTSR-rec.pt',
                    help='Path to pre-trained TTSR weights')
parser.add_argument('--freeze_lte', type=str2bool, default=False,
                    help='Freeze LTE (VGG) weights during training')
parser.add_argument('--freeze_stages', type=str, default='sfe,stage1,stage2',
                    help='Comma-separated MainNet stages to freeze: sfe,stage1,stage2,stage3,merge')
parser.add_argument('--ref_correction', type=str2bool, default=True,
                    help='Add an always-on lightweight reference correction head before LTE')
parser.add_argument('--ref_correction_feats', type=int, default=16,
                    help='Channel width of the reference correction head')
parser.add_argument('--no_reference', type=str2bool, default=False,
                    help='Disable EVERY reference-dependent path (texture search/transfer, '
                         'reference illumination transfer and the correction head) to train or '
                         'evaluate a no-reference baseline')
parser.add_argument('--ref_correct_w', type=float, default=0.0,
                    help='Weight for auxiliary L1 loss between corrected ref and GT. 0 disables it')
parser.add_argument('--eval_mean_align', type=str2bool, default=True,
                    help='Also report mean-aligned metrics (prediction rescaled to GT Y-mean) '
                         'alongside the raw metrics; diagnostic only, never used for checkpoint selection')
parser.add_argument('--illum_match_w', type=float, default=0.0,
                    help='Weight for low-frequency illumination/tone matching loss against GT')
parser.add_argument('--illum_match_factor', type=int, default=8,
                    help='Average-pool factor for the illumination matching loss')
parser.add_argument('--ref_illum_pool', type=int, default=8,
                    help='Average-pool factor of the reference illumination transfer (low-frequency scale)')
parser.add_argument('--no_ref_illum', type=str2bool, default=False,
                    help='Disable the reference illumination transfer (RefIllumTransfer). The module is '
                         'still constructed so the weight initialisation stays bit-identical to a run '
                         'that keeps it; only its contribution to the output is removed.')
parser.add_argument('--ref_illum_const_ref', type=str2bool, default=False,
                    help='CONTROL EXPERIMENT: keep RefIllumTransfer fully intact (same capacity, same '
                         'training) but feed it a constant (all-zero) image instead of the reference. '
                         'Separates "extra low-frequency correction capacity" from "reference content".')
parser.add_argument('--oracle_matching', type=str, default='off',
                    choices=['off', 'index', 'full'],
                    help='DIAGNOSTIC ONLY, never for deployment: replace the learned patch '
                         'matching with the ground-truth correspondence (identity). Requires '
                         'the reference to be the aligned high-light image. "index" swaps only '
                         'the transferred content, "full" also swaps the similarity that '
                         'becomes the gate S so that T and S stay consistent.')
parser.add_argument('--eval_chroma_gain', type=float, default=1.0,
                    help='Scale chroma (Cb/Cr about neutral) of the evaluated prediction; '
                         '1.0 disables. Fixed colour calibration, never uses GT')
parser.add_argument('--eval_tta', type=str2bool, default=False,
                    help='Test-time augmentation at eval: average 4 rotations x 2 flips (8 views)')
parser.add_argument('--eval_data1', type=str2bool, default=False,
                    help='Also evaluate data1 during training validation')
parser.add_argument('--eval_lol_nanobanana', type=str2bool, default=False,
                    help='Also evaluate LOLv1 nanobanana refs during training validation')
parser.add_argument('--lol_nanobanana_eval_ref_dir', type=str,
                    default='/root/data/datasets/LOLdataset/eval15/nanobanana_ref',
                    help='LOLv1 clean nanobanana ref directory for eval')
parser.add_argument('--eval_lolv2_nanobanana', type=str2bool, default=False,
                    help='Also evaluate lolv2 nanobanana refs during lolv2real/syn training')
parser.add_argument('--eval_lolv2real_gt', type=str2bool, default=False,
                    help='Also evaluate lolv2real GT during mixed_lolv2_data1 training')
parser.add_argument('--data1_eval_dir', type=str, default='/root/data/datasets/data1',
                    help='Dataset directory for extra data1 evaluation')
parser.add_argument('--mixed_data1_dir', type=str, default='/root/data/datasets/data1',
                    help='data1 directory used by mixed_lolv2_data1 dataset')
parser.add_argument('--mixed_data1_nanobanana_dir', type=str, default='/root/data/datasets/data1',
                    help='data1 directory used by mixed_data1_nanobanana_lolv2 dataset')
parser.add_argument('--mixed_lolv2_real_dir', type=str, default='/root/data/datasets/lol-v2-real',
                    help='lolv2real directory used by mixed_lolv2_data1 dataset')
parser.add_argument('--mixed_lolv2_syn_dir', type=str, default='/root/data/datasets/lol-v2-synthetic',
                    help='lolv2syn directory used by mixed_lolv2_data1 dataset')
parser.add_argument('--mixed_weights', type=str, default='2,1,1',
                    help='Sampling weights for data1,lolv2real,lolv2syn in mixed dataset')
parser.add_argument('--mixed_weights_4', type=str, default='2,3,1,1',
                    help='Sampling weights for data1_gt,data1_nanobanana,lolv2real,lolv2syn')
parser.add_argument('--nanobanana_ref_subdir', type=str, default='nanobanana_ref',
                    help='Subdirectory name for nanobanana refs inside data1 camera folders')
parser.add_argument('--train_manifest_dir', type=str, default='',
                    help='Restrict data1 TrainSet to the samples listed in '
                         '<dir>/<camera>.txt (matched-subset ablation). Empty = full set')
parser.add_argument('--nanobanana_manifest_dir', type=str,
                    default='/root/data/datasets/data1/.nanobanana_sample_manifest',
                    help='Directory containing Huawei.txt / Nikon.txt manifest files for the 250-sample subset')
parser.add_argument('--data1_nanobanana_eval_ref_dir', type=str, default='',
                    help='Optional external clean nanobanana ref directory for data1 eval')
parser.add_argument('--data1_nanobanana_eval_huawei_ref_dir', type=str,
                    default='/root/data/datasets/data1/Eval/Huawei/nanobanana_ref',
                    help='Huawei clean nanobanana ref directory for data1 eval')
parser.add_argument('--data1_nanobanana_eval_nikon_ref_dir', type=str,
                    default='/root/data/datasets/data1/Eval/Nikon/nanobanana_ref_Nikon',
                    help='Nikon clean nanobanana ref directory for data1 eval')
parser.add_argument('--lolv2_nanobanana_subset', type=str, default='real',
                    help='Subset for lolv2 nanobanana training: real / syn')
parser.add_argument('--lolv2_nanobanana_ref_subdir', type=str, default='nanobanana_ref_v2',
                    help='Subdirectory name for lolv2 nanobanana refs under Train/')
parser.add_argument('--lolv2_nanobanana_mixed_weights', type=str, default='1,1',
                    help='Sampling weights for lolv2real_nanobanana,lolv2syn_nanobanana')
parser.add_argument('--mixed_data1_lolv2_nanobanana_weights', type=str,
                    default='3,3,1,2',
                    help='Sampling weights for data1_gt,data1_nanobanana,real_nanobanana,syn_nanobanana')
parser.add_argument('--lolv2_nanobanana_eval_real_ref_dir', type=str,
                    default='/root/data/datasets/lol-v2-real/Test/nanobanana_ref_v3',
                    help='lolv2real clean nanobanana ref directory for eval')
parser.add_argument('--lolv2_nanobanana_eval_syn_ref_dir', type=str,
                    default='/root/data/datasets/lol-v2-synthetic/Test/nanobanana_ref_v3',
                    help='lolv2syn clean nanobanana ref directory for eval')
parser.add_argument('--data1_psnr_floor', type=float, default=20.4,
                    help='Early-stop threshold for data1 PSNR during training')
parser.add_argument('--data1_nanobanana_huawei_floor', type=float, default=20.4,
                    help='Early-stop threshold for data1 Huawei nanobanana PSNR')
parser.add_argument('--data1_nanobanana_nikon_floor', type=float, default=17.0,
                    help='Early-stop threshold for data1 Nikon nanobanana PSNR')
parser.add_argument('--early_stop_on_data1', type=str2bool, default=False,
                    help='Stop training if data1 PSNR falls below data1_psnr_floor')

### loss setting
parser.add_argument('--GAN_type', type=str, default='WGAN_GP',
                    help='The type of GAN used in training')
parser.add_argument('--GAN_k', type=int, default=2,
                    help='Training discriminator k times when training generator once')
parser.add_argument('--tpl_use_S', type=str2bool, default=False,
                    help='Whether to multiply soft-attention map in transferal perceptual loss')
parser.add_argument('--tpl_type', type=str, default='l2',
                    help='Which loss type to calculate gram matrix difference in transferal perceptual loss [l1 / l2]')
parser.add_argument('--rec_w', type=float, default=1.,
                    help='The weight of reconstruction loss')
parser.add_argument('--rec_loss_type', type=str, default='l1',
                    help='Reconstruction loss type: l1 / l2 / charbonnier')
parser.add_argument('--per_w', type=float, default=0.1,
                    help='The weight of perceptual loss')
parser.add_argument('--tpl_w', type=float, default=0.1,
                    help='The weight of transferal perceptual loss')
parser.add_argument('--adv_w', type=float, default=0.,
                    help='The weight of adversarial loss')
### low-light specific losses
parser.add_argument('--illum_smooth_w', type=float, default=1.0,
                    help='Weight of illumination smoothness loss (TV regularization)')
parser.add_argument('--color_w', type=float, default=0.5,
                    help='Weight of color constancy loss (Grey-World)')
parser.add_argument('--exposure_w', type=float, default=1.0,
                    help='Weight of exposure control loss')

### optimizer setting
parser.add_argument('--beta1', type=float, default=0.9,
                    help='The beta1 in Adam optimizer')
parser.add_argument('--beta2', type=float, default=0.999,
                    help='The beta2 in Adam optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='The eps in Adam optimizer')
parser.add_argument('--lr_rate', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--lr_rate_dis', type=float, default=1e-4,
                    help='Learning rate of discriminator')
parser.add_argument('--lr_rate_lte', type=float, default=1e-5,
                    help='Learning rate of LTE')
parser.add_argument('--lr_rate_stage2', type=float, default=-1,
                    help='Separate learning rate for Stage2; -1 means use lr_rate')
parser.add_argument('--lr_rate_refhead', type=float, default=-1,
                    help='Separate learning rate for the reference correction head; -1 means use lr_rate')
parser.add_argument('--lr_rate_illum', type=float, default=-1,
                    help='Separate learning rate for the global illumination head; -1 means use lr_rate')
parser.add_argument('--lr_rate_refillum', type=float, default=-1,
                    help='Separate learning rate for the reference illumination transfer module; -1 means use lr_rate')
parser.add_argument('--decay', type=float, default=999999,
                    help='Learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5,
                    help='Learning rate decay factor for step decay')

### training setting
parser.add_argument('--batch_size', type=int, default=4,
                    help='Training batch size (smaller for larger crop)')
parser.add_argument('--train_crop_size', type=int, default=128,
                    help='Training data crop size (larger for low-light context)')
parser.add_argument('--num_init_epochs', type=int, default=5,
                    help='The number of init epochs which are trained with only reconstruction loss')
parser.add_argument('--num_epochs', type=int, default=50,
                    help='The number of training epochs')
parser.add_argument('--print_every', type=int, default=10,
                    help='Print period')
parser.add_argument('--save_every', type=int, default=5,
                    help='Save period')
parser.add_argument('--val_every', type=int, default=5,
                    help='Validation period')

### evaluate / test / finetune setting
parser.add_argument('--eval', type=str2bool, default=False,
                    help='Evaluation mode')
parser.add_argument('--eval_save_results', type=str2bool, default=False,
                    help='Save each image during evaluation')
parser.add_argument('--tile_size', type=int, default=256,
                    help='Tile size used by tiled inference for large images')
parser.add_argument('--tile_overlap', type=int, default=96,
                    help='Overlap between adjacent tiles during tiled inference')
parser.add_argument('--tile_window', type=str, default='cosine',
                    help='Blending window for tiled inference: linear / cosine')
parser.add_argument('--model_path', type=str, default=None,
                    help='The path of model to evaluation')
parser.add_argument('--test', type=str2bool, default=False,
                    help='Test mode')
parser.add_argument('--lr_path', type=str, default='./test/demo/lr/lr.png',
                    help='The path of input low-light image when testing')
parser.add_argument('--ref_path', type=str, default='./test/demo/ref/ref.png',
                    help='The path of ref image when testing')

args = parser.parse_args()
