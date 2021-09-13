import os.path as osp

###########################################################################
############################## PLEASE EDIT ################################
###########################################################################

# FOLDER WHERE YOU STORE YOUR DATASETS
DS_DIR = ''

# path to COCO, MPI-INF-3DHP, and 3DPW dataset (see SPIN datasets)
COCO_ROOT = f'/{DS_DIR}/SPINDS/datasets/COCO'
MPI_INF_3DHP_ROOT = f'/{DS_DIR}/SPINDS/datasets/mpi_inf_3dhp'
PW3D_ROOT = f'/{DS_DIR}/SPINDS/datasets/3DPW'
# MTP dataset root
MTP_ROOT = f'/{DS_DIR}/tuch/mtp/release/mtp'
# path to TUCH DSC annotations folder
DSC_ROOT = f'/{DS_DIR}/tuch/dsc/release'
# path to the folders with the original images
DF_ROOT = f'/{DS_DIR}/tuch/dsc/images/df'
LSP_ROOT = f'/{DS_DIR}/tuch/dsc/images/lsp'
LSPET_ROOT = f'/{DS_DIR}/tuch/dsc/images/lspet'
# After running EFT on all images, you can set EFT_FOLDER
EFT_PATH = f'/{DS_DIR}/eft/eft_fit'

###########################################################################
################################# DONE #####################################
############################################################################



###########################################################################
########################## PLEASE DO NOT EDIT #############################
###########################################################################

DBS_PATH = 'data/dbs'
DATASET_FILES = {
    'train': 
    {
        'mpi-inf-3dhp': osp.join(DBS_PATH, 'mpi_inf_3dhp_train.pt'),
        'dsc_df': osp.join(DBS_PATH, 'dsc_df_train.pt'),
        'dsc_lspet': osp.join(DBS_PATH, 'dsc_lspet_train.pt'),
        'dsc_lsp': osp.join(DBS_PATH, 'dsc_lsp_train.pt'),
        'mtp': osp.join(DBS_PATH, 'mtp_train.pt'),
        '3dpw': osp.join(DBS_PATH, '3dpw_train.pt'),
        'dsc_df_eft': osp.join(DBS_PATH, 'dsc_df_eft_train.pt'),
        'dsc_lspet_eft': osp.join(DBS_PATH, 'dsc_lspet_eft_train.pt'),
        'dsc_lsp_eft': osp.join(DBS_PATH, 'dsc_lsp_eft_train.pt'),
    },
    'val': 
    { 
        'mtp': osp.join(DBS_PATH, 'mtp_val.pt')
    },
    'test': 
    {
        'mpi-inf-3dhp': osp.join(DBS_PATH, 'mpi_inf_3dhp_test.pt'),
        '3dpw': osp.join(DBS_PATH, '3dpw_test.pt')
    }
}

IMAGE_FOLDERS = {'mpi-inf-3dhp': MPI_INF_3DHP_ROOT,
                   'coco': COCO_ROOT,
                   'coco_eft': COCO_ROOT,
                   '3dpw': PW3D_ROOT,
                   'dsc_df': osp.join(DF_ROOT, 'images'),
                   'dsc_lspet': osp.join(LSPET_ROOT, 'images'),
                   'dsc_lsp': osp.join(LSP_ROOT, 'images'),
                   'dsc_df_eft': osp.join(DF_ROOT, 'images'),
                   'dsc_lspet_eft': osp.join(LSPET_ROOT, 'images'),
                   'dsc_lsp_eft': osp.join(LSP_ROOT, 'images'),
                   'mtp': osp.join(MTP_ROOT, 'images')
                }

# If you followed the install instructions, you can keep the paths below as the are
SMPL_MODEL_DIR = 'data/models/smpl'
SMPLX_MODEL_DIR = 'data/models/smplx'
SMPL_MODELS_DIR = 'data'
JOINT_REGRESSOR_TRAIN_EXTRA = 'data/essentials/spin/J_regressor_extra.npy'
JOINT_REGRESSOR_H36M = 'data/essentials/spin/J_regressor_h36m.npy'
STATIC_FITS_DIR = 'data/static_fits'
SMPL_MEAN_PARAMS = 'data/essentials/spin/smpl_mean_params.npz'
PRIOR_FOLDER = 'data/essentials/spin'
THREEDPW_CIG = 'data/essentials/3dpw_test_csig_pc.npy'
SMPLX_TO_SMPL = 'data/essentials/models_utils/smplx_to_smpl.pkl'
SPIN_MODEL_CHECKPOINT = 'data/spin_model_checkpoint.pt'
GEODESICS_SMPL = 'data/essentials/geodesics/smpl/smpl_neutral_geodesic_dist.npy'
HD_MODEL_DIR = 'data/essentials/hd_model/smpl'
SEGMENT_DIR = 'data/essentials/segments/smpl'

# Default variables for contact
geothres = 0.3
euclthres = 0.02