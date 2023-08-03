You should be able to take the train/val data point to it within normal_train.py
from there run sub_normal on 1 matador node with 2 gpu -c 4 and possibly add in the sbatch command for more memory per cpu if needed.
Once this ahs completed you will have a model with x epochs output and then send your best model into the pred_photon pred_pion files and that will run a prediction on our standard testing files.
from there, pick one of the comparison plot programs and get it linked up with the predictions you made and you should be able to create the same plots i do.
 once everything works properly, You can begin modifiying the normal_train.py file with additional flags or modifiers to see if it causes improvements. next, you can slightly play wiith tf_model_keras.py file with notes from jordan damgov's email that i will forward to you.
 good luck
data is:
/lustre/research/hep/hgcdpg/amankel/gnn/3D-Data/
GNN-3D-150Gev-train_csv_800k.pkl 
GNN-3D-150Gev-val_csv_800k.pkl
GNN-3D-150Gev_photon.pkl (for testing only)
GNN-3D-150Gev_pion.pkl (for testing only)

first 2 files are 900k events split into 70% train 30% validation
last 2 files are 100k Events for photon/pion this is our standard testing set and shouldnt be used for training/validation
