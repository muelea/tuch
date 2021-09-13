import joblib
import glob
import os.path as osp
import numpy as np

"""
Script that loads the lsp dataset and replaces the pose and shape 
parameter with the eft fits. Then saves a new .pt file to data/dbs.
E.g. useful when you fit EFT on the cluster and want to merge the 
output of each cluster GPU into a single .pt file.
"""

new_dataset_names = ['tucheft'] #['eft', 'tucheft']
datasets = [['dsc_lsp_train', 'dsc_lsp'], ['dsc_lspet_train', 'dsc_lspet'],
            ['dsc_df_train','dsc_df']]

def merge_temps(new_name, dsnames):
    dsname, dsnameshort = dsnames
    cbs = 10
    dsoutname = dsname.replace('_train', '_{}_train'.format(new_name))

    dbpath = 'data/dbs/{}.pt'.format(dsname)
    dboutpath = 'data/dbs/{}.pt'.format(dsoutname)
    data = joblib.load(dbpath)
    data['betas'] = np.zeros((len(data['imgname']), 10))
    data['pose'] = np.zeros((len(data['imgname']), 72))

    tempfiles = glob.glob(osp.join('temp', new_name, dsnameshort, '*'))
    print('Num temp files available ', len(tempfiles))
    totalcount=0
    print([osp.join('temp', new_name, dsnameshort,
                    dsname.replace('_train', '_{}_train'.format(new_name)) + '_' + str(idx) + '.pt')
           for idx in range(len(tempfiles)) if not osp.exists(osp.join('temp', new_name, dsnameshort,
                    dsname.replace('_train', '_{}_train'.format(new_name)) + '_' + str(idx) + '.pt'))])

    for idx in range(len(tempfiles)):
        try:
            tempdata = joblib.load(osp.join('temp', new_name, dsnameshort,
                    dsname.replace('_train', '_{}_train'.format(new_name)) + '_' + str(idx) + '.pt'))
        except:
            print('files doe not exist ', osp.join('temp', new_name, dsnameshort,
                    dsname.replace('_train', '_{}_train'.format(new_name)) + '_' + str(idx) + '.pt'))
            continue
        tempdataidx = np.where(abs(tempdata['pose'].sum(1)) > 0)[0]
        if len(tempdataidx) != cbs:
            print(len(tempdataidx), osp.join('temp', new_name, dsnameshort,
                    dsname.replace('_train', '_{}_train'.format(new_name)) + '_' + str(idx) + '.pt'))
        totalcount += len(tempdataidx)
        data['betas'][tempdataidx, :] = tempdata['betas'][tempdataidx, :]
        data['pose'][tempdataidx, :] = tempdata['pose'][tempdataidx, :]

    print('Toal fitted poses ', totalcount, 'from', len(data['imgname']))
    print('save merged files to: ', dboutpath)
    joblib.dump(data, dboutpath)


for new_name in new_dataset_names:
    for dataset in datasets:
        print('process ', new_name, dataset)
        merge_temps(new_name, dataset)