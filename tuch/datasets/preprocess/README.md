## Data preparation
Besides the demo code, we also provide training and evaluation code for our approach. To use this functionality, you need to download the relevant datasets.

### Training Data
1. **MTP** Download the MTP Data from our [website](https://tuch.is.tue.mpg.de) to MTP_ROOT. 
Then, edit the ```configs/config.py``` file with the ```${MTP_ROOT}``` path.

2. **DSC** Download the discrete self-contact annotations from our [website](https://tuch.is.tue.mpg.de). Unzip the folder to DSC_ROOT.
In this folder you find three subfolders (`DSC_ROOT/df`, `DSC_ROOT/lsp`, `DSC_ROOT/lspet`) that contain the self-contact annotations. A `regions` folder with the SMPL and SMPL-X vertex ids per region. And a file called tuch_bodypart_pairs, which has the regions pairs we use in TUCH training.

- You also need to download the images for [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html). Download low resolution images and keypoint annotations for [LSP](https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose.html) and [LSP Extended](https://dbcollection.readthedocs.io/en/latest/datasets/leeds_sports_pose_extended.html):
```
${DF_ROOT}
|-- images
    |-- img_00000001.jpg
    |-- img_00000002.jpg
```
```
${LSP_ROOT}
|-- images
    |-- im0001.jpg
    |-- im0002.jpg
joints.mat
```
```
${LSPET_ROOT}
|-- images
    |-- im00001.png
    |-- im00002.png
joints.mat
```

Then, edit the ```configs/config.py``` file with the ```${DSC_ROOT} ${DF_ROOT}, ${LSPET_ROOT}, and ${LSP_ROOT}``` path.


### Test Data
3. **3DPW**: You need to download the data from the [dataset website](https://virtualhumans.mpi-inf.mpg.de/3DPW/). After you unzip the dataset files, please complete the root path of the dataset in the file ```config.py```.

4. **MPI-INF-3DHP**: Again, we use this dataset for training and evaluation. You need to download the data from the [dataset website](http://gvv.mpi-inf.mpg.de/3dhp-dataset). The expected fodler structure at the end of the processing looks like:
```
${MPI_INF_3DHP_ROOT}
|-- mpi_inf_3dhp_test_set
    |-- TS1
|-- S1
    |-- Seq1
        |-- imageFrames
            |-- video_0
```
Then, you need to edit the ```config.py``` file with the ```${MPI_INF_3DHP_ROOT}``` path.

Due to the large size of this dataset we subsample the frames used by a factor of 10. Also, in the training .npz files, we have included fits produced by SMPL fitting on 2D joints predictions from multiple views. Since, the quality of these fits is not perfect, we only keep 60% of the fits fixed, while the rest are updated within the typical SPIN loop.

### Generate dataset files
Then generate the training files:
```
python preprocess_datasets.py --train_files_tuch --val_files_tuch --test_files_tuch
```
