# Prostate-Classification-MIL
The `AMIL.py` file uses attention based MIL on pure-patches i.e. the patches which are entirely belonging to a particular class. The `No_AMIL.py` uses a finetuning of ResNet18 to do the same classification. The `Pure_patches_Data` file has the link to the pure patches data set with train and test split.  

The `MIL Attempt1.py` attempts to use all the patches in the dataset to train. The `data_prep_for_MIL.py` file containes the code to make patches of desired size for each WSI to make bags and store them.
