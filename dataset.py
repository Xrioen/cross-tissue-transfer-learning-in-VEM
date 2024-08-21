# from dataclasses import replace
import numpy as np
import torch
# import tifffile as tff
# import matplotlib.pyplot as plt
import glob

    

class SNEMI3D_Tiled_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_SNEMI3D"
        self.ID1 = sorted(glob.glob(self.loc + "/X_train_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_train/*"))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y

    
class SNEMI3D_Tiled_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_SNEMI3D"
        self.ID1 = sorted(glob.glob(self.loc + "/X_test_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_test/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y



class Urocell_Tiled_fold1_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_frac = 1, subsample_number = 0, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-0-0-0" not in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-0-0-0" not in x]

        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
            
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y


class Urocell_Tiled_fold1_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-0-0-0" in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-0-0-0" in x]

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
        
class Urocell_Tiled_fold2_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_frac = 1, subsample_number = 0, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-3-2-1" not in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-3-2-1" not in x]

        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y


class Urocell_Tiled_fold2_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-3-2-1" in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-3-2-1" in x]

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
        
class Urocell_Tiled_fold3_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_frac = 1, subsample_number = 0, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-4-3-0" not in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-4-3-0" not in x]

        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y


class Urocell_Tiled_fold3_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_urocell"
        self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        self.ID1 = [x for x in self.ID1 if "fib1-4-3-0" in x]
        self.ID2 = sorted(glob.glob(self.loc + "/Y_all/*"))
        self.ID2 = [x for x in self.ID2 if "fib1-4-3-0" in x]

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y




    
    

    
class HarvardLiverDataset_LD_Tiled_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_train_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_train_LD/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
            
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
class HarvardLiverDataset_LD_Tiled_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_test_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_test_LD/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
        
        
class HarvardLiverDataset_ER_Tiled_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_train_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_train_ER/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
class HarvardLiverDataset_ER_Tiled_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_test_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_test_ER/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y        

class HarvardLiverDataset_mito_Tiled_train_CLAHE(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_train_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_train_mito/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
            
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
class HarvardLiverDataset_mito_Tiled_test_CLAHE(torch.utils.data.Dataset):
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_harvard_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_test_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_test_mito/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y



######################################################################


class RatLiverDataset_Tiled_all(torch.utils.data.Dataset):
    def __init__(self, take_first = -1):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        
        if take_first == -1:
            self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))
        else:
            self.ID1 = sorted(glob.glob(self.loc + "/X_all_CLAHE/*"))[0:take_first]

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        
        'Generates one sample of data'
        return self.re_X
        


class RatLiverDataset_Tiled_Train18(torch.utils.data.Dataset):
    #mito
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_train_mito_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_train_mito/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
    
    
class RatLiverDataset_Tiled_Test18(torch.utils.data.Dataset):
    #mito
    def __init__(self):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_test_mito_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_test_mito/*"))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y


class RatLiverDataset_Tiled_ER_18_35(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_ER_18_35_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_ER_18_35/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y

class RatLiverDataset_Tiled_ER_18_35_train(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_ER_18_35_train_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_ER_18_35_train/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y

class RatLiverDataset_Tiled_ER_18_35_test(torch.utils.data.Dataset):
    def __init__(self, subsample_number = 0, subsample_frac = 1, seed = 27):
        self.loc = "/home/xrioen/scratch/tiled_rat_liver"
        self.ID1 = sorted(glob.glob(self.loc + "/X_ER_18_35_test_CLAHE/*"))
        self.ID2 = sorted(glob.glob(self.loc + "/Y_ER_18_35_test/*"))
        
        if subsample_frac != 1:
            np.random.seed(seed)
            n = len(self.ID1)
            ind1 = np.random.choice(np.arange(n), size = int(np.floor(subsample_frac*n)), replace = False)
            self.ID1 = np.array(self.ID1)[ind1]
            self.ID2 = np.array(self.ID2)[ind1]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))
        
        if subsample_number != 0:
            np.random.seed(seed)
            n = len(self.ID1)
            ind2 = np.random.choice(np.arange(n), size = subsample_number, replace = False)
            self.ID1 = np.array(self.ID1)[ind2]
            self.ID2 = np.array(self.ID2)[ind2]
            print(str(len(self.ID1)) + " patches selected out of " + str(n))

    def __len__(self):
        return (len(self.ID1))

    def __getitem__(self, index):
        self.re_X = torch.load(self.ID1[index])
        self.re_Y = torch.load(self.ID2[index])
        
        'Generates one sample of data'
        return self.re_X, self.re_Y
    