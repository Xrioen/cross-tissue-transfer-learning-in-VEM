#Infer using pretrained network trained on SNEMI3D


import torch
import torch.nn as nn
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob
import tifffile as tff
from tqdm import tqdm
from rsunet import RSUNet
from dataset import RatLiverDataset_Tiled_all



def infer(X, model_loc, output_location, X_rev = None,
          output_dim = (54,2000,3000), fov = (18,160,160), 
          stride = (18,160,160), save_images_b = False, cutoff = 127.5, pred_weights = np.ones((18,160,160))):
    
        #X_rev is tiled raw data starting fro the opposite corner in 3d
        #pred_weights is a matrix specifying the weight of the prediction at each pixel. Make edges lower weight may improve performance
        #output_dim is the image size after taking into account any downsampling during preprocessing
        #stride equivalent to the one used in tile_datasets
        
        assert (fov == (18,160,160))
        patch_dim1 = (output_dim[2] - fov[2] + stride[2])//stride[2]  # patch_dim1 correspond to the number of patches along x
        patch_dim2 = (output_dim[1] - fov[1] + stride[1])//stride[1]  # patch_dim2 correspond to the number of patches along Y
        s1 = stride[2]
        s2 = stride[0]

        model = RSUNet().cuda()
        try:
            model.load_state_dict(torch.load(str(model_loc))) #entire model is pretrained and loaded
        except: 
            print("layer names do not match, truncating names and retrying")
            #when models only share some layers, load only shared layers
#             state_dict2 = torch.load("Exp_19_2/trained_model_baseline_r0.61_t0.65.pth")
            state_dict2 = torch.load(str(model_loc))

#             for key in list(state_dict2)[:2]:
#                 del state_dict2[key]
#             for key in list(state_dict2)[-2:]:
#                 del state_dict2[key]
            #use above in case names are the same but corresponding to different layer in the two models
            for key in list(state_dict2):
                new_key = str(key).split(".")[1:]
                new_key = ".".join(new_key)

                state_dict2[new_key] = state_dict2.pop(key)

            pretrained_dict = state_dict2
            model_dict = model.state_dict()

            #filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            model_dict.update(pretrained_dict)  #overwrite entries in the existing state dict
            model.load_state_dict(model_dict)  #load the new state dict
        
        model.eval()
        
        loader = torch.utils.data.DataLoader(X, batch_size=1, shuffle=False)

        output_stacked = np.zeros((len(loader),1,18,160,160))
        output_unstacked = np.zeros(output_dim)
        gt_unstacked = np.zeros(output_dim)
        overlap_mtx = np.zeros(output_dim)
        
        for value, data in enumerate(tqdm(loader)):
            with torch.no_grad():
                sig0 = nn.Sigmoid()
                pred = model(data.cuda())
                output = sig0(pred)
                output_stacked[value, :, :, :, :] = output.detach().cpu().numpy()
        
        output_stack1 = output_stacked.squeeze() #1440x18x160x160 
        output_stack1 = output_stack1.reshape((-1, patch_dim2, patch_dim1,18,160,160))

        for z in range(output_stack1.shape[0]):
            for y in range(output_stack1.shape[1]):
                for x in range(output_stack1.shape[2]):
                    output_unstacked[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += output_stack1[z,y,x,:,:,:] * pred_weights
                    #gt_unstacked[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += gt_stack1[z,y,x,:,:,:]
                    overlap_mtx[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += pred_weights
        
        
        if X_rev != None:
            #X_rev is the raw input images patched starting from the opposite end
            loader = torch.utils.data.DataLoader(X_rev, batch_size=1, shuffle=False)

            output_stacked = np.zeros((len(loader),1,18,160,160))
    

            for value, data in enumerate(tqdm(loader)):
                with torch.no_grad():
                    sig0 = nn.Sigmoid()
                    pred = model(data.cuda())
                    output = sig0(pred)
                    output_stacked[value, :, :, :, :] = output.detach().cpu().numpy()
            
            output_stack1 = output_stacked.squeeze() #1440x18x160x160 
            output_stack1 = output_stack1.reshape((-1, patch_dim2, patch_dim1,18,160,160))
            
            output_unstacked = np.flip(output_unstacked)
            gt_unstacked = np.flip(gt_unstacked)
            overlap_mtx = np.flip(overlap_mtx)
            
            for z in range(output_stack1.shape[0]):
                for y in range(output_stack1.shape[1]):
                    for x in range(output_stack1.shape[2]):
                        output_unstacked[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += output_stack1[z,y,x,:,:,:] * pred_weights
                        #gt_unstacked[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += gt_stack1[z,y,x,:,:,:]
                        overlap_mtx[z*s2:(z*s2+18),y*s1:(y*s1+160),x*s1:(x*s1+160)] += pred_weights
                        
            output_unstacked = np.flip(output_unstacked)
            gt_unstacked = np.flip(gt_unstacked)
            overlap_mtx = np.flip(overlap_mtx)
        
        
        #final_gt = gt_unstacked/overlap_mtx
        overlap_mtx[overlap_mtx == 0] = 1
        final_output = output_unstacked/overlap_mtx*255
        
        final_output_binary = np.where(final_output > cutoff, 255, 0)
        
        
        
        
        plt.imshow(final_output[10,:,:], cmap = "gray")
        plt.figure()
        plt.imshow(final_output_binary[10,:,:], cmap = "gray")
#         print(final_output.shape())
        
        
        
        if save_images_b == True:
            if not os.path.isdir(output_location):
                os.mkdir(output_location)
            np.save(output_location+"/final_output.npy", final_output)
            for z in range(0, final_output.shape[0], final_output.shape[0]//5):
                cv2.imwrite(output_location + "/predict_borders_z" + str("%.3d" %z) + ".png", final_output[z , : , :])
                cv2.imwrite(output_location + "/predict_borders_binary_z" + str("%.3d" %z) + ".png", final_output_binary[z , : , :])
#             for x in range(10):
#                 cv2.imwrite(output_location + "/predict_borders_x" + str("%.3d" %x) + ".png", final_output[: , : , x+100])
        return(final_output)


#Example of ER inference on rat liver data
data = RatLiverDataset_Tiled_all()
output_dim = (219,2000,2000)
stride = (9,80,80)
model_loc = "finetuned_ER_model.pth"
output_location = "rat_liver_ER_inference"

output = infer(data, model_loc, output_location, 
          output_dim = output_dim,  
          stride = stride, save_images_b = True)

for z in range(0, output.shape[0], output.shape[0]//5):
    alpha = 0.5
    img = tff.imread('/home/xrioen/scratch/EM20-324-200_series_processed/' + 
                                                str(z+1).zfill(3) + "____z" + str(z) + '.0.tif')
    print(img.shape)
    pred = cv2.resize(output[z,:,:], (7000,7000), interpolation = cv2.INTER_CUBIC)
    
    merged = cv2.merge([img, img, img])
    merged[:,:,0][pred > 127.5] = merged[:,:,0][pred > 127.5]*alpha + 255*(1-alpha)
    merged[:,:,1][pred > 127.5] = merged[:,:,0][pred > 127.5]*alpha
    merged[:,:,2][pred > 127.5] = merged[:,:,0][pred > 127.5]*alpha
#     plt.imshow(merged)
    plt.imsave(output_location + "/predict_borders_merged_z" + str("%.3d" %z) + ".png", merged)
    
    
