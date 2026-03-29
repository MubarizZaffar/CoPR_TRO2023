import os
from os import path as osp
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
from service.benchmark_base import Benchmark
from dataset import SevenScenesRelPoseQueryDataset, SevenScenesRelPoseRefSingleSequenceDataset, SevenScenesRelPoseRefExtrapolatedDataset, SevenScenesRelPoseRefEPLRDataset
from augmentations import get_augmentations
from model import RelPoseNetOrg, COPR
import pickle
from scipy.spatial.transform import Rotation as R
import faiss
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
import cv2
import time
import matplotlib.image as mpimg
from PIL import Image

densification_time=0 # time spent to densify the map
retrieval_time_sparse=0 #time spent on retrieving the match for a single query from a sparse map
retrieval_time_dense=0
encoding_time=0 #encoding time for a single query image
matching_time_sparse=0 #time needed to match a query descriptor with reference descriptors in original/sparse map
matching_time_dense=0
dense_map_size=0 #size of extrapolated map 
sparse_map_size=0 #size of original sparse map

def loss(feat_gt, feat):
    
        loss = nn.MSELoss()
        scale=1000
        pose_loss = loss(torch.as_tensor(feat_gt), torch.as_tensor(feat)) * scale        
        return pose_loss

class SevenScenesRetrievalTest(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader_ref, self.dataloader_ref_ep, self.dataloader_ref_eplr, self.dataloader_query, self.ref_dataset, self.ref_dataset_ep,\
        self.ref_dataset_eplr, self.query_dataset, self.queryimages_poses, self.refimages_poses, self.refimages_ep_poses = self._init_dataloader()
        
        self.modelrelposenetorg = self._load_model_RelPoseNetOrg().to(self.device)
        self.modelcopr = self._load_model_COPR().to(self.device)
        
    def _init_dataloader(self):
        experiment_cfg = self.cfg.experiment_params

        # define test augmentations
        train_augs, eval_aug = get_augmentations()

        # test data
        ref_dataset = SevenScenesRelPoseRefSingleSequenceDataset(self.cfg, transforms=eval_aug)
        ref_dataset_ep = SevenScenesRelPoseRefExtrapolatedDataset(self.cfg, transforms=eval_aug)
        ref_dataset_eplr = SevenScenesRelPoseRefEPLRDataset(self.cfg, transforms=eval_aug)
        query_dataset = SevenScenesRelPoseQueryDataset(self.cfg, transforms=eval_aug)
        
        # pose information
        queryimages_poses, refimages_poses = ref_dataset._get_gt_poses() # Get the ground-truth poses for queries, references and reference images extended/regressed/COPR'ed
        refimages_ep_poses = ref_dataset_ep._get_gt_poses()
        
        # define a dataloader
        dataloader_ref = torch.utils.data.DataLoader(ref_dataset,
                                                 batch_size=self.cfg.experiment_params.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=False)

        # define a dataloader
        dataloader_refep = torch.utils.data.DataLoader(ref_dataset_ep,
                                                 batch_size=self.cfg.experiment_params.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=False)
        
        # define a dataloader
        dataloader_refeplr = torch.utils.data.DataLoader(ref_dataset_eplr,
                                                 batch_size=self.cfg.experiment_params.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=False)
        # define a dataloader
        dataloader_query = torch.utils.data.DataLoader(query_dataset,
                                                 batch_size=self.cfg.experiment_params.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=8,
                                                 drop_last=False)

        return dataloader_ref, dataloader_refep, dataloader_refeplr, dataloader_query, ref_dataset, ref_dataset_ep, ref_dataset_eplr, query_dataset, queryimages_poses, refimages_poses, refimages_ep_poses

    def _load_model_RelPoseNetOrg(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg
        
        modelrelposenetorg = RelPoseNetOrg(model_params_cfg)
        
        if (self.cfg.experiment_params.loss_type == 'relativepose'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenetorg.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with relative pose... Done!')

        elif (self.cfg.experiment_params.loss_type == 'triplet'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenettriplet.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with triplet... Done!')

        elif (self.cfg.experiment_params.loss_type == 'distance'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenetdistancebased.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with distance based... Done!')

        return modelrelposenetorg.eval()
    
    def _load_model_COPR(self):
        print(f'Loading COPR model...')
        model_params_cfg = self.cfg
        
        modelcopr = COPR(model_params_cfg)

        data_dict = torch.load(model_params_cfg.model_paramsCOPR.snapshot)
        modelcopr.load_state_dict(data_dict['state_dict'])
        print(f'Loading COPR model... Done!')
        return modelcopr.eval()
    
    def _get_rotation_diff(self, r_qpose, r_bestmatchpose):
        
        mag1=np.linalg.norm(r_qpose)
        mag2=np.linalg.norm(r_bestmatchpose)
        
        r_qpose=r_qpose/mag1
        r_bestmatchpose=r_bestmatchpose/mag2
        dp=abs(np.dot(r_qpose,r_bestmatchpose))

        if (dp<=1):
            angle_diff=2*math.acos(dp)*(180.0/math.pi)
        else:
            angle_diff=0 # Multiplication precision can lead to values slightly higher than 1, which would give math domain error
        
        return angle_diff
    
    def get_lrmodelparam(self,anchor1feat, anchor2feat, anchor3feat, anchor4feat, tpose1, tpose2, tpose3, tpose4, _dim, _no_of_samples, _fdim):    
        dim=_dim #dimensionality of the problem 3D in this case
        no_of_samples=_no_of_samples #samples to fit a hyper-plane 
        fdim=_fdim #dimension of feature vector
        Y_samples=np.concatenate((anchor1feat, anchor2feat, anchor3feat, anchor4feat), axis=0).reshape(no_of_samples, fdim)  # 4x512 i.e. No. of samples x feature dimension
        X_samples=np.concatenate((tpose1, tpose2, tpose3, tpose4), axis=0).reshape(no_of_samples, dim) # 4x3 i.e. No. of samples x problem dimensions (1D, 2D, 3D)
        X_samples_hg=np.insert(X_samples, X_samples.shape[1], values=1, axis=1) # homogenous coordinates for bias

        if(np.linalg.det(X_samples_hg.T @ X_samples_hg)!=0):
            # print('I had a good set of tposes')
            params = (np.linalg.inv(X_samples_hg.T @ X_samples_hg) @ X_samples_hg.T @ Y_samples)
     
        else:
            print('Singularity occured for tpose '+ str(tpose1) + ' ' + str(tpose2) + ' ' + str(tpose3) + ' ' + str(tpose4) + ' , adding some small random noise: ')
        
        return params
    
    def get_desc(self,lrmodel, tposereg):
        tposereg_hg=np.insert(tposereg, tposereg.shape[0], values=1, axis=0) # homogenous coordinates for bias
        regdesc=lrmodel.T @ tposereg_hg
        
        return regdesc
        
    def postprocess_lr_regdesc(self, regressed_desc,pca, pptype):
        if(pptype=='whitening'):        
            whitened = pca.fit_transform(regressed_desc)
            whitened[whitened<0] = 0
            return whitened
        
        elif(pptype=='norm'):        
            normdesc=regressed_desc/np.linalg.norm(regressed_desc)
            normdesc[normdesc<0] = 0
            return normdesc

        elif(pptype=='logit'):
            regressed_declogit=np.zeros(len(regressed_desc))
            for itr,val in enumerate(regressed_desc):
                regressed_declogit[itr]=1/(1+np.exp(-1*val))
                                           
            return regressed_declogit
        
        elif(pptype=='none'):
            return regressed_desc

        else:
            print('Unknown post processing type')

    
    def get_color_basedoneuc(self,euc):
        if (self.cfg.experiment_params.datasetname=='7scenes'):
            if (euc<=0.3):
                color='lime'
                
            elif (euc>0.3 and euc<=0.5):
                color='darkorange'
                
            elif (euc>0.5):
                color='red'

        elif (self.cfg.experiment_params.datasetname=='University'):
            if (euc<=1):
                color='cyan'
                
            elif (euc>1 and euc<=2):
                color='orange'
                
            elif (euc>2):
                color='red'
                
        elif (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
            if (euc<=1):
                color='cyan'
                
            elif (euc>1 and euc<=2):
                color='orange'
                
            elif (euc>2):
                color='red'
                
        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            if (euc<=1.5):
                color='lime'
                
            elif (euc>1.5 and euc<=2.5):
                color='darkorange'
                
            elif (euc>2.5):
                color='red'        
                
        return color 
    
    def visualize_matches(self, matches_org, matches_ep, matches_eplr, matches_sanitycheck, queryimages_poses, refimages_ep_poses, refimages_poses, gt_matches_org, gt_matches_ep):      

        if (self.cfg.experiment_params.datasetname=='7scenes'):
            subsampling=3
            anchor_sampling=40 #40 for 7scenes, 1 for escalator

        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            subsampling=2
            anchor_sampling=1 #40 for 7scenes, 1 for escalator
            
        refimages_poses_x_list=[]
        refimages_poses_y_list=[]
        refimages_poses_z_list=[]
        
        refimages_poses_x_list_anchor=[]
        refimages_poses_y_list_anchor=[]
        refimages_poses_z_list_anchor=[]

        for itr, val in enumerate(refimages_poses):
            refimages_poses_x_list.append(val[0])
            refimages_poses_y_list.append(val[1])
            refimages_poses_z_list.append(val[2])
            
            if (itr%anchor_sampling==0):
                refimages_poses_x_list_anchor.append(val[0])
                refimages_poses_y_list_anchor.append(val[1])
                refimages_poses_z_list_anchor.append(val[2])
        
        refimages_ep_x_list=[]
        refimages_ep_y_list=[]
        refimages_ep_z_list=[]
        for itr, val in enumerate(refimages_ep_poses):          
            refimages_ep_x_list.append(val[0])
            refimages_ep_y_list.append(val[1])   
            refimages_ep_z_list.append(val[2])   
        
        query_x_list=[]
        query_y_list=[]
        query_z_list=[]
        
        for itr, val in enumerate(queryimages_poses):
            query_x_list.append(val[0])
            query_y_list.append(val[1])  
            query_z_list.append(val[2])  
                   
        matches_org_x_list=[]
        matches_org_y_list=[]
        matches_org_z_list=[]
        for itr, match in enumerate(matches_org):
            match=int(match)
            matches_org_x_list.append(refimages_poses[match][0])
            matches_org_y_list.append(refimages_poses[match][1])
            matches_org_z_list.append(refimages_poses[match][2])
               
        matches_ep_x_list=[]
        matches_ep_y_list=[]
        matches_ep_z_list=[]
        for itr, match in enumerate(matches_ep):
            match=int(match)
            matches_ep_x_list.append(refimages_ep_poses[match][0])
            matches_ep_y_list.append(refimages_ep_poses[match][1])
            matches_ep_z_list.append(refimages_ep_poses[match][2])
            
        matches_eplr_x_list=[]
        matches_eplr_y_list=[]
        matches_eplr_z_list=[]

        for itr, match in enumerate(matches_eplr):
            match=int(match)
            matches_eplr_x_list.append(refimages_ep_poses[match][0])
            matches_eplr_y_list.append(refimages_ep_poses[match][1])
            matches_eplr_z_list.append(refimages_ep_poses[match][2])

        matches_sanitycheck_x_list=[]
        matches_sanitycheck_y_list=[]
        matches_sanitycheck_z_list=[]

        for itr, match in enumerate(matches_sanitycheck):
            match=int(match)
            matches_sanitycheck_x_list.append(refimages_ep_poses[match][0])
            matches_sanitycheck_y_list.append(refimages_ep_poses[match][1])
            matches_sanitycheck_z_list.append(refimages_ep_poses[match][2])
            
        gt_matches_org_x_list=[]
        gt_matches_org_y_list=[]
        gt_matches_org_z_list=[]

        for itr, match in enumerate(gt_matches_org):
            match=int(match)
            gt_matches_org_x_list.append(refimages_poses[match][0])
            gt_matches_org_y_list.append(refimages_poses[match][1])
            gt_matches_org_z_list.append(refimages_poses[match][2])
               
        gt_matches_ep_x_list=[]
        gt_matches_ep_y_list=[]
        gt_matches_ep_z_list=[]

        for itr, match in enumerate(gt_matches_ep):
            match=int(match)
            gt_matches_ep_x_list.append(refimages_ep_poses[match][0])
            gt_matches_ep_y_list.append(refimages_ep_poses[match][1])
            gt_matches_ep_z_list.append(refimages_ep_poses[match][2])

        print('visualizing matches')
    
        if (self.cfg.experiment_params.datasetname=='7scenes'):
            plt.figure(figsize=(14, 5)) #10,5
            plt.subplot(141)
            plt.xlabel('x (meters)') 
            plt.ylabel('y (meters)')   
            plt.title("a) $M_{sparse}$")
            plt.plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='white', markersize=0.9)  # just to keep the plot scale same with other subplots
            plt.plot(query_x_list, query_y_list, '.', color='black', label='Query');        
            plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='fuchsia', label='Ref');
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_org_x_list, matches_org_y_list, '.', color='orange');  
            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1], val[2]])-np.array([matches_org_x_list[itr],matches_org_y_list[itr], matches_org_z_list[itr]]))
                    plt.plot([val[0],matches_org_x_list[itr]],[val[1],matches_org_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            plt.legend(loc=4)
    
            plt.subplot(142)
            plt.xlabel('x (meters)') 
            # plt.ylabel('meters')   
            plt.title("b) $M_{dense}$")
            plt.plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='dodgerblue', label='Ref. Extrapolated', markersize=0.9); 
            plt.plot(query_x_list, query_y_list, '.', color='black', label='Query')   
            plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='fuchsia', label='Ref. non-anchors');  
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_ss_x_list, matches_ss_y_list, '.', color='orange');  
            # for itr, val in enumerate(queryimages_poses):
            #     if (itr%subsampling==0): #subsampling here for better visualization
            #         euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_eplr_x_list[itr],matches_eplr_y_list[itr],matches_eplr_z_list[itr]]))
            #         plt.plot([val[0],matches_eplr_x_list[itr]],[val[1],matches_eplr_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=0.5,markersize=0.7, alpha=0.18)
            plt.plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='gold', label='Ref. anchors')  
    
            plt.legend(loc=4)
    
            
            plt.subplot(143)
            plt.xlabel('x (meters)') 
            # plt.ylabel('meters')   
            plt.title("c) $M_{dense}$ (Lin. Reg.)")
            plt.plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='white', markersize=0.9); 
            plt.plot(query_x_list, query_y_list, '.', color='black', label='Query')   
            # plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='pink', label='Ref. non-anchors');  
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_ss_x_list, matches_ss_y_list, '.', color='orange');  
            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_eplr_x_list[itr],matches_eplr_y_list[itr],matches_eplr_z_list[itr]]))
                    plt.plot([val[0],matches_eplr_x_list[itr]],[val[1],matches_eplr_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            # plt.plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='red', label='Ref. anchors')  
    
            plt.legend(loc=4)
            
            plt.subplot(144)
            plt.xlabel('x (meters)') 
            # plt.ylabel('meters')   
            # plt.title("Median Translation Error: 0.28 meters", fontsize=16)
            plt.title("d) $M_{dense}$ (Non-lin. Reg.)")
             
            plt.plot(refimages_ep_x_list, refimages_ep_y_list, 'x', color='white', markersize=0.9)  
            plt.plot(query_x_list, query_y_list, '.', color='black', label='Query') 
            # plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='pink', label='Ref. non-anchors');  
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_ss_x_list, matches_ss_y_list, '.', color='orange');  
            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_ep_x_list[itr],matches_ep_y_list[itr],matches_ep_z_list[itr]]))
                    plt.plot([val[0],matches_ep_x_list[itr]],[val[1],matches_ep_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            # plt.plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='red', label='Ref. anchors'); 
            plt.legend(loc=4)
            
            # plt.savefig('/home/mzaffar/Documents/COPR_stuff/office_extrapolation_v2_dpi800.png', dpi=800, bbox_inches='tight')
            plt.show()

        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            
            fig, ax = plt.subplots(1, 4, figsize=(15, 5), gridspec_kw={'width_ratios': [1.5, 1, 1, 1]})
            
            # plt.figure(figsize=(15, 5)) #10,5
            
            img = Image.open('/home/mzaffar/Documents/COPR_stuff/escalator_query_ref_sample.png')
            # img.thumbnail((512, 512))
            
            # plt.subplot(141)
            ax[0].imshow(img)
            ax[0].axis('off')
            ax[0].set_title("a) Example Images")
            
            # plt.subplot(142)
            ax[1].set_xlabel('x (meters)') 
            ax[1].set_ylabel('y (meters)')   
            ax[1].set_title("b) $M_{sparse}$")
            ax[1].plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='white', markersize=0.9)  # just to keep the plot scale same with other subplots
            ax[1].plot(query_x_list, query_y_list, '.', color='black', label='Query');        
            ax[1].plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='fuchsia', label='Ref');
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_org_x_list, matches_org_y_list, '.', color='orange');  
            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1], val[2]])-np.array([matches_org_x_list[itr],matches_org_y_list[itr], matches_org_z_list[itr]]))
                    ax[1].plot([val[0],matches_org_x_list[itr]],[val[1],matches_org_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            ax[1].legend(loc=4)
                
            # plt.subplot(143)
            ax[2].set_xlabel('x (meters)') 
            # plt.ylabel('meters')   
            ax[2].set_title("c) $M_{dense}$ (Lin. Reg.)")
            ax[2].plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='white', markersize=0.9); 
            ax[2].plot(query_x_list, query_y_list, '.', color='black', label='Query')   
            ax[2].plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='dodgerblue', label='Ref. Extrapolated')#, markersize=0.9); 
            # plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='pink', label='Ref. non-anchors');  
            
            # plt.plot(refimages_ss_x_list, refimages_ss_y_list, '.', color='black');  
            # plt.plot(matches_ss_x_list, matches_ss_y_list, '.', color='orange');  
            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_eplr_x_list[itr],matches_eplr_y_list[itr],matches_eplr_z_list[itr]]))
                    ax[2].plot([val[0],matches_eplr_x_list[itr]],[val[1],matches_eplr_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            ax[2].plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='gold', label='Ref. anchors')  
            ax[2].legend(loc=4)
            
            # ax[3].subplot(144)
            ax[3].set_xlabel('x (meters)') 
            # plt.ylabel('meters')   
            # plt.title("Median Translation Error: 0.28 meters", fontsize=16)
            ax[3].set_title("d) $M_{dense}$ (Non-lin. Reg.)")
             
            ax[3].plot(refimages_ep_x_list, refimages_ep_y_list, 'x', color='white', markersize=0.9)  
            ax[3].plot(query_x_list, query_y_list, '.', color='black', label='Query') 
            ax[3].plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='dodgerblue', label='Ref. Extrapolated')#, markersize=0.9); 

            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_ep_x_list[itr],matches_ep_y_list[itr],matches_ep_z_list[itr]]))
                    ax[3].plot([val[0],matches_ep_x_list[itr]],[val[1],matches_ep_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            ax[3].plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='gold', label='Ref. anchors'); 
            ax[3].legend(loc=4)
            
            axin = ax[3].inset_axes([0.05, 0.6, 0.35, 0.35])
            
            axin.plot(refimages_ep_x_list, refimages_ep_y_list, 'x', color='white', markersize=0.9)  
            axin.plot(query_x_list, query_y_list, '.', color='black', label='Query') 
            axin.plot(refimages_ep_x_list, refimages_ep_y_list, '.', color='dodgerblue', label='Ref. Extrapolated')#, markersize=0.9); 

            for itr, val in enumerate(queryimages_poses):
                if (itr%subsampling==0): #subsampling here for better visualization
                    euc=np.linalg.norm(np.array([val[0],val[1],val[2]])-np.array([matches_ep_x_list[itr],matches_ep_y_list[itr],matches_ep_z_list[itr]]))
                    axin.plot([val[0],matches_ep_x_list[itr]],[val[1],matches_ep_y_list[itr]],'o-',color=self.get_color_basedoneuc(euc),linewidth=1.0,markersize=0.7, alpha=0.35)
            axin.plot(refimages_poses_x_list_anchor, refimages_poses_y_list_anchor, 'x', color='gold', label='Ref. anchors'); 
            
            axin.set_xlim(-0.25, 0.55)
            axin.set_ylim(1.5, 2.1)
            axin.set_xticks([])
            axin.set_yticks([])
            
            ax[3].indicate_inset_zoom(axin)
            
            # plt.savefig('/home/mzaffar/Documents/COPR_stuff/escalator_results_v2_dpi600.png', dpi=600, bbox_inches='tight')
            plt.show()

    def visualize_coprplusrelpose_matches(self, qindex, querypose, refposes, retrievedpose, retrplusregpose, maptype):      
        refimages_poses_x_list=[]
        refimages_poses_y_list=[]
        refimages_poses_z_list=[]
        
        for itr, val in enumerate(refposes):
            refimages_poses_x_list.append(val[0])
            refimages_poses_y_list.append(val[1])
            refimages_poses_z_list.append(val[2])

        plt.plot(querypose[0],querypose[1],'.', color='green', label="Query", markersize=10.0)
        plt.plot(retrievedpose[0],retrievedpose[1],'x', color='orange', label="Retrieval Pose", markersize=10.0)
        plt.plot(retrplusregpose[0],retrplusregpose[1],'x', color='red', label="Ret. + Reg. Pose", markersize=10.0)
        plt.plot(refimages_poses_x_list, refimages_poses_y_list, '.', color='blue', label='Reference Points');  
        
        plt.legend(loc=0)
        # plt.show()
        
        path=self.cfg.paths.workdir+'/coprplusrelposeplots/'+self.cfg.experiment_params.datasetname+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
        if not os.path.exists(path):
            os.makedirs(path)

        savepath=path+str(qindex)+'.jpg'
        plt.savefig(savepath) 
        plt.clf()

    def show_qualitative_matches(self,itr, match, gt_bestmatch, sequence):    
                              
            data_dir=self.cfg.paths.dataset_path + self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]
            q_name=data_dir+'/'+self.query_dataset.fnames_query[itr]
            
            if (match<len(self.ref_dataset.fnames_ref)): # because the last ref pose in relposefailure experiment is the query pose, and the last image (i.e. regressed image) is essentially the query image 
                bm_name=data_dir+'/'+self.ref_dataset.fnames_ref[match]

            else:
                bm_name=q_name
                
            q_image=cv2.imread(q_name)
            bm_image=cv2.imread(bm_name)
                
            combined=np.concatenate((q_image,bm_image),axis=1)
    
            cv2.namedWindow(sequence,cv2.WINDOW_NORMAL)
            cv2.imshow(sequence, combined)         

    def evaluate(self):
        total_ref_images = self.ref_dataset.__len__()
        total_ref_images_ep = self.ref_dataset_ep.__len__()
        total_query_images = self.query_dataset.__len__()
        
        sparse_map_size = total_ref_images
        dense_map_size = total_ref_images_ep
        
        print('total_ref_images: ', total_ref_images) # for logging
        print('total_query_images: ', total_query_images) # for logging
        
        descs_ref_org = np.empty((total_ref_images, self.cfg.model_paramsCOPR.features_dim))
        descs_query = np.empty((total_query_images, self.cfg.model_paramsCOPR.features_dim))
        
        descs_ref_ep = np.empty((total_ref_images_ep, self.cfg.model_paramsCOPR.features_dim))
        descs_ref_sanitycheck = np.empty((total_ref_images_ep, self.cfg.model_paramsCOPR.features_dim))
        descs_ref_eplr = np.empty((total_ref_images_ep, self.cfg.model_paramsCOPR.features_dim))
        
        t_diff_org = 0
        t_diff_org_list = []
        t_diff_org_rpose_list = []
        t_diff_ep = 0
        t_diff_ep_list = []
        t_diff_ep_rpose_list = []
        t_diff_eplr = 0
        t_diff_eplr_list = []
        t_diff_sanitycheck = 0
        t_diff_sanitycheck_list = []
        
        r_diff_org = 0
        r_diff_org_list = []
        r_diff_ep = 0
        r_diff_ep_list = []
        r_diff_eplr = 0
        r_diff_eplr_list = []
        r_diff_sanitycheck = 0
        r_diff_sanitycheck_list = []

        gt_t_diff_org_avg = 0
        gt_t_diff_org_list = []
        gt_t_diff_ep_avg = 0
        gt_t_diff_ep_list = []
        gt_t_diff_eplr_avg = 0
        gt_t_diff_eplr_list = []

        gt_r_diff_org_avg = 0
        gt_r_diff_org_list = []
        gt_r_diff_ep_avg = 0
        gt_r_diff_ep_list = []
        gt_r_diff_eplr_avg = 0
        gt_r_diff_eplr_list = []
        

        with torch.no_grad():
            itr=0
            pca = PCA(whiten=True)
            
            if (self.cfg.experiment_params.coprplusrelposeexp==False): #Can't do linear regression with the very sparse anchors of copr+relpose experiment
                if (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
                    # For extr
                    _dim=2
                    _no_of_samples=4 
                    _fdim=512
    
                else:
                    _dim=3
                    _no_of_samples=4 
                    _fdim=512
                    
                if (self.cfg.experiment_params.descriptors_stored==False):
                    for data_batch in tqdm(self.dataloader_ref_eplr):  #extrapolation using least-squares linear regression
                        # print('index:', data_batch['index'])
                        if (data_batch['requires_regression']==False):
                            feat1, _ , _ , _ = self.modelrelposenetorg(data_batch['anchor1img'].to(self.device),
                                                  data_batch['anchor1img'].to(self.device))
                            descs_ref_eplr[data_batch['index'],:]=feat1.cpu().numpy()
                            # print('No need to regress for me')
                        
                        else:    
                            anchor1feat, _ , _ , _ = self.modelrelposenetorg(data_batch['anchor1img'].to(self.device), \
                                                      data_batch['anchor1img'].to(self.device))
                            anchor1feat=anchor1feat.cpu().numpy().reshape(self.cfg.model_paramsCOPR.features_dim,1)
                            
                            anchor2feat, _ , _ , _ = self.modelrelposenetorg(data_batch['anchor2img'].to(self.device), \
                                                      data_batch['anchor2img'].to(self.device))
                            anchor2feat=anchor2feat.cpu().numpy().reshape(self.cfg.model_paramsCOPR.features_dim,1)
                            
                            anchor3feat, _ , _ , _ = self.modelrelposenetorg(data_batch['anchor3img'].to(self.device), \
                                                      data_batch['anchor3img'].to(self.device))
                            anchor3feat=anchor3feat.cpu().numpy().reshape(self.cfg.model_paramsCOPR.features_dim,1)
        
                            anchor4feat, _ , _ , _ = self.modelrelposenetorg(data_batch['anchor4img'].to(self.device), \
                                                      data_batch['anchor4img'].to(self.device))
                            anchor4feat=anchor4feat.cpu().numpy().reshape(self.cfg.model_paramsCOPR.features_dim,1)                    
                            
                            tpose1=np.asarray(data_batch['tpose1']).reshape(_dim,1)
                            tpose2=np.asarray(data_batch['tpose2']).reshape(_dim,1)
                            tpose3=np.asarray(data_batch['tpose3']).reshape(_dim,1)
                            tpose4=np.asarray(data_batch['tpose4']).reshape(_dim,1)
                            
                            tposereg=np.asarray(data_batch['tposereg']).reshape(_dim,1)
                            
                            lrmodel=self.get_lrmodelparam(anchor1feat, anchor2feat, anchor3feat, anchor4feat, tpose1, tpose2, tpose3, tpose4, _dim, _no_of_samples, _fdim)
                            
                            # regressed_desc_pp = 0.6 * anchor2feat + 0.4 * anchor3feat  # just for testing relposefailure exp
                            regressed_desc = self.get_desc(lrmodel, tposereg) 
                            regressed_desc_pp = self.postprocess_lr_regdesc(regressed_desc,pca, pptype='none')
                            
                            # print('regresseddesc',regressed_desc_pp)
                            # print('anchor1feat',anchor1feat)
                        
                            descs_ref_eplr[data_batch['index'],:]=regressed_desc_pp.squeeze()
                            
                    maptype = 'eplr'
                    path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname + '/' + self.cfg.experiment_params.loss_type +'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
                    if not os.path.exists(path):
                        os.makedirs(path)
                    
                    with open(path+'descs.npy', 'wb') as f:
                        np.save(path+'descs.npy', descs_ref_eplr)
                            
                else:
                    maptype = 'eplr'
                    path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'

                    with open(path+'descs.npy', 'rb') as f:
                        descs_ref_eplr = np.load(f)
                    
            # print(descs_ref_eplr)
            print('Computed all extrapolated reference descriptors using original RelPoseNet and linear least squares regression') 
            
            if (self.cfg.experiment_params.descriptors_stored==False):
                for data_batch in tqdm(self.dataloader_ref):
                    # print('ISnaity check')
                    feat1, _ , _ , _ = self.modelrelposenetorg(data_batch['img'].to(self.device),
                                              data_batch['img'].to(self.device))
                    
                    descs_ref_org[data_batch['index'],:]=feat1.cpu().numpy()
                    
                maptype = 'original'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                with open(path+'descs.npy', 'wb') as f:
                    np.save(path+'descs.npy', descs_ref_org)
           
            else:
                maptype = 'original'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'

                with open(path+'descs.npy', 'rb') as f:
                    descs_ref_org = np.load(f)
                
            print('Computed all reference descriptors using original RelPoseNet')
            
            
            if (self.cfg.experiment_params.descriptors_stored==False):
                for data_batch in tqdm(self.dataloader_ref_ep):  #For uniform subsampling this could be easily subsampled using original ref descriptors but dataloader provided in case there is non-uniform sub-sampling required
                    
                    if (data_batch['requires_regression']==False):
                        feat1, _ , _ , _ = self.modelrelposenetorg(data_batch['img'].to(self.device),
                                              data_batch['img'].to(self.device))
                        descs_ref_ep[data_batch['index'],:]=feat1.cpu().numpy() 
                    
                    else:    
                        anchorfeat, _ , _ , _ = self.modelrelposenetorg(data_batch['img'].to(self.device),
                                                  data_batch['img'].to(self.device))
                     
                        regression_time = time.time()
                        regressed_desc = self.modelcopr(torch.squeeze(anchorfeat), \
                                                  torch.squeeze(data_batch['rel_pose']).to(self.device)) 
        
                        #Just to test if original ep descriptors are helpful or not 
                        # regressed_desc, _ , _ , _ = self.modelrelposenetorg(data_batch['regimg'].to(self.device),
                        #                           data_batch['regimg'].to(self.device))
            
                        ############
                        
                        descs_ref_ep[data_batch['index'],:]=regressed_desc.cpu().numpy()
                        regression_time = time.time() - regression_time
                        
                        global densification_time
                        densification_time += regression_time
                        
                maptype = 'CoPR'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                with open(path+'descs.npy', 'wb') as f:
                    np.save(path+'descs.npy', descs_ref_ep)
            else:
                maptype = 'CoPR'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'

                with open(path+'descs.npy', 'rb') as f:
                    descs_ref_ep = np.load(f)

            print('Computed all extrapolated reference descriptors using original RelPoseNet and COPR')    

            if (self.cfg.experiment_params.descriptors_stored==False):            
                for data_batch in tqdm(self.dataloader_ref_ep):  #For uniform subsampling this could be easily subsampled using original ref descriptors but dataloader provided in case there is non-uniform sub-sampling required
                    
                    if (data_batch['requires_regression']==False):
                        feat1, _ , _ , _ = self.modelrelposenetorg(data_batch['img'].to(self.device),
                                              data_batch['img'].to(self.device))
                        descs_ref_sanitycheck[data_batch['index'],:]=feat1.cpu().numpy()
                    
                    else:
                        
                        descs_ref_sanitycheck[data_batch['index'],:]=np.random.rand(self.cfg.model_paramsCOPR.features_dim)
                
                maptype = 'sanitycheck'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                with open(path+'descs.npy', 'wb') as f:
                    np.save(path+'descs.npy', descs_ref_sanitycheck)   
            else:     
                maptype = 'sanitycheck'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'

                with open(path+'descs.npy', 'rb') as f:
                    descs_ref_sanitycheck = np.load(f)
                
                
            print('Computed all sanity check reference descriptors')  
            
            encoding_time = 0
            if (self.cfg.experiment_params.descriptors_stored==False):            
                encoding_time = time.time()
                for data_batch in tqdm(self.dataloader_query):
                    feat1, _ , _ , _ = self.modelrelposenetorg(data_batch['img'].to(self.device),
                                              data_batch['img'].to(self.device))
                    
                    descs_query[data_batch['index'],:] = feat1.cpu().numpy()
                    
    
                encoding_time = time.time() - encoding_time
                encoding_time = encoding_time / total_query_images             
            
                maptype = 'query'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'
                if not os.path.exists(path):
                    os.makedirs(path)
                    
                with open(path+'descs.npy', 'wb') as f:
                    np.save(path+'descs.npy', descs_query) 
            else:
                maptype = 'query'
                path = self.cfg.paths.workdir+'/saveddescriptors/'+self.cfg.experiment_params.datasetname+ '/' + self.cfg.experiment_params.loss_type+'/'+self.ref_dataset.scenes_dict[self.cfg.experiment_params.chosen_scene]+'/'+maptype+'/'

                with open(path+'descs.npy', 'rb') as f:
                    descs_query = np.load(f)
                    
            print('Computed all query descriptors using original RelPoseNet')          
            
            # Computing matches for original version: Upper bound
            matching_time_sparse = time.time()
            faiss_index = faiss.IndexFlatL2(self.cfg.model_paramsCOPR.features_dim)
            faiss_index.add(descs_ref_org.astype('float32'))
            _, matches_org = faiss_index.search(descs_query.astype('float32'), 1) # Top-1
            matching_time_sparse = (time.time() - matching_time_sparse) / total_query_images
            retrieval_time_sparse = encoding_time + matching_time_sparse
            scores, matches_org_Top5 = faiss_index.search(descs_query.astype('float32'), 5) # Top-5
            # print('Original Matches Top 5: ',scores,matches_org_Top5)
            
            
            # Computing matches for upsampled version: EP COPR
            matching_time_dense = time.time()
            faiss_index = faiss.IndexFlatL2(self.cfg.model_paramsCOPR.features_dim)
            faiss_index.add(descs_ref_ep.astype('float32'))
            _, matches_ep = faiss_index.search(descs_query.astype('float32'), 1) # Top-1
            matching_time_dense = (time.time() - matching_time_dense) / total_query_images
            retrieval_time_dense = encoding_time + matching_time_dense
            scores, matches_ep_Top5 = faiss_index.search(descs_query.astype('float32'), 6) # Top-5
            # print('EP Matches Top 5: ',scores,matches_ep_Top5)
            
            
            # Computing matches for upsampled version: Sanity Check
            faiss_index = faiss.IndexFlatL2(self.cfg.model_paramsCOPR.features_dim)
            faiss_index.add(descs_ref_sanitycheck.astype('float32'))
            _, matches_sanitycheck = faiss_index.search(descs_query.astype('float32'), 1) # Top-1

            if (self.cfg.experiment_params.coprplusrelposeexp==False):
                # Computing matches for upsampled version: EP Linear Regressions
                faiss_index = faiss.IndexFlatL2(self.cfg.model_paramsCOPR.features_dim)
                faiss_index.add(descs_ref_eplr.astype('float32'))
                _, matches_eplr = faiss_index.search(descs_query.astype('float32'), 1) # Top-1
                scores, matches_eplr_Top5 = faiss_index.search(descs_query.astype('float32'), 6) # Top-5
                # print('EPLR Matches Top 5: ',scores,matches_eplr_Top5)
            

            # Computing ground-truth matches for original version
            faiss_index = faiss.IndexFlatL2(3)
            faiss_index.add(np.asarray(self.refimages_poses)[:,0:3].astype('float32'))
            _, gt_matches_org = faiss_index.search(np.asarray(self.queryimages_poses)[:,0:3].astype('float32'), 1) # Top-1
            
            # Computing groubd-truth matches for subsampled version
            faiss_index = faiss.IndexFlatL2(3)
            faiss_index.add(np.asarray(self.refimages_ep_poses)[:,0:3].astype('float32'))
            _, gt_matches_ep = faiss_index.search(np.asarray(self.queryimages_poses)[:,0:3].astype('float32'), 1) # Top-1
            
            for itr, match in enumerate(matches_org):
                # print('Original Matches: ',itr, match)
                match=int(match)
                t_qpose =  self.queryimages_poses[itr][0:3]              
                t_bestmatchpose =  self.refimages_poses[match][0:3]              
                diff_t = np.linalg.norm(t_qpose-t_bestmatchpose)            
                
                r_qpose =  self.queryimages_poses[itr][3::]   # quaternion           
                r_bestmatchpose =  self.refimages_poses[match][3::]    # quaternion    
                # print('r_qpose',r_qpose)
                # print('r_bestmatchpose',r_bestmatchpose)
                
                diff_r = self._get_rotation_diff(r_qpose, r_bestmatchpose)   # This takes as an imput quaternions, converts them to rotation vectors, computes the relative rotation between them and returns the absolute angle error            

                t_diff_org = t_diff_org + diff_t
                r_diff_org = r_diff_org + diff_r
                
                t_diff_org_list.append(diff_t)
                r_diff_org_list.append(diff_r)
                
                gt_bestmatch=int(gt_matches_org[itr])
                gt_t_bestmatchpose = self.refimages_poses[gt_bestmatch][0:3]
                gt_diff_t = np.linalg.norm(t_qpose-gt_t_bestmatchpose)
                gt_t_diff_org_list.append(gt_diff_t)
                gt_t_diff_org_avg=gt_t_diff_org_avg+gt_diff_t
                gt_r_bestmatchpose=self.refimages_poses[gt_bestmatch][3:7]
                gt_diff_r=self._get_rotation_diff(r_qpose, gt_r_bestmatchpose)
                gt_r_diff_org_avg=gt_r_diff_org_avg+gt_diff_r
                gt_r_diff_org_list.append(gt_diff_r)
                
                desc_q = torch.from_numpy(descs_query[itr]).float()
                desc_ref = torch.from_numpy(descs_ref_org[match]).float()
                r_pose_q, r_pose_t = self.modelrelposenetorg.forward_relpose(desc_q.to(self.device),desc_ref.to(self.device))
                r_pose_q=r_pose_q.cpu().numpy()
                r_pose_t=r_pose_t.cpu().numpy()
                
                t_estimate=t_bestmatchpose-r_pose_t
                diff_t_withrpose=np.linalg.norm(t_qpose-t_estimate) 
                t_diff_org_rpose_list.append(diff_t_withrpose)

                # self.show_qualitative_matches(itr, match, gt_bestmatch, 'Org Office')                
                # self.visualize_coprplusrelpose_matches(itr,t_qpose, self.refimages_poses, t_bestmatchpose, t_estimate, 'Original_map')
                
            print('errors computed for original DB') 
                
            for itr, match in enumerate(matches_ep):
                # print(itr, match)
                match=int(match)
                t_qpose =  self.queryimages_poses[itr][0:3]              
                t_bestmatchpose =  self.refimages_ep_poses[match][0:3]              
                diff_t = np.linalg.norm(t_qpose-t_bestmatchpose)             
                
                r_qpose =  self.queryimages_poses[itr][3::]   # quaternion           
                r_bestmatchpose =  self.refimages_ep_poses[match][3::]    # quaternion    
                # print('r_qpose',r_qpose)
                # print('r_bestmatchpose',r_bestmatchpose)
                
                diff_r = self._get_rotation_diff(r_qpose, r_bestmatchpose)   # This takes as an input quaternions, converts them to rotation vectors, computes the relative rotation between them and returns the absolute angle error            

                t_diff_ep = t_diff_ep + diff_t
                r_diff_ep = r_diff_ep + diff_r 
                
                t_diff_ep_list.append(diff_t)
                r_diff_ep_list.append(diff_r)
                
                gt_bestmatch=int(gt_matches_ep[itr])
                gt_t_bestmatchpose = self.refimages_ep_poses[gt_bestmatch][0:3]
                gt_diff_t = np.linalg.norm(t_qpose-gt_t_bestmatchpose)
                gt_t_diff_ep_list.append(gt_diff_t)
                gt_t_diff_ep_avg=gt_t_diff_ep_avg+gt_diff_t
                gt_r_bestmatchpose=self.refimages_ep_poses[gt_bestmatch][3:7]
                # print(r_qpose,gt_r_bestmatchpose)
                gt_diff_r=self._get_rotation_diff(r_qpose, gt_r_bestmatchpose)
                gt_r_diff_ep_avg=gt_r_diff_ep_avg+gt_diff_r
                gt_r_diff_ep_list.append(gt_diff_r)
                
                desc_q = torch.from_numpy(descs_query[itr]).float()
                desc_ref = torch.from_numpy(descs_ref_ep[match]).float()
                r_pose_q, r_pose_t = self.modelrelposenetorg.forward_relpose(desc_q.to(self.device),desc_ref.to(self.device))
                r_pose_q=r_pose_q.cpu().numpy()
                r_pose_t=r_pose_t.cpu().numpy()
                
                t_estimate=t_bestmatchpose-r_pose_t
                diff_t_withrpose=np.linalg.norm(t_qpose-t_estimate) 
                t_diff_ep_rpose_list.append(diff_t_withrpose)

                # self.show_qualitative_matches(itr, match, gt_bestmatch, 'EP Office')                    
                # self.visualize_coprplusrelpose_matches(itr,t_qpose, self.refimages_ep_poses, t_bestmatchpose, t_estimate, 'COPR_map')
                
            print('errors computed for COPR upsampled DB')
            
            for itr, match in enumerate(matches_sanitycheck):
                # print(itr, match)
                match=int(match)
                t_qpose =  self.queryimages_poses[itr][0:3]              
                t_bestmatchpose =  self.refimages_ep_poses[match][0:3]              
                diff_t = np.linalg.norm(t_qpose-t_bestmatchpose)             
                
                r_qpose =  self.queryimages_poses[itr][3::]   # quaternion           
                r_bestmatchpose =  self.refimages_ep_poses[match][3::]    # quaternion    
                # print('r_qpose',r_qpose)
                # print('r_bestmatchpose',r_bestmatchpose)
                
                diff_r = self._get_rotation_diff(r_qpose, r_bestmatchpose)   # This takes as an imput quaternions, converts them to rotation vectors, computes the relative rotation between them and returns the absolute angle error            

                t_diff_sanitycheck = t_diff_sanitycheck + diff_t
                r_diff_sanitycheck = r_diff_sanitycheck + diff_r 
                
                t_diff_sanitycheck_list.append(diff_t)
                r_diff_sanitycheck_list.append(diff_r)
                                
            print('errors computed for sanity check')

            if (self.cfg.experiment_params.coprplusrelposeexp==False):
                for itr, match in enumerate(matches_eplr):
                    # print(itr, match)
                    match=int(match)
                    t_qpose =  self.queryimages_poses[itr][0:3]              
                    t_bestmatchpose =  self.refimages_ep_poses[match][0:3]              
                    diff_t = np.linalg.norm(t_qpose-t_bestmatchpose)             
                    
                    r_qpose =  self.queryimages_poses[itr][3::]   # quaternion           
                    r_bestmatchpose =  self.refimages_ep_poses[match][3::]    # quaternion    
                    # print('r_qpose',r_qpose)
                    # print('r_bestmatchpose',r_bestmatchpose)
                    
                    diff_r = self._get_rotation_diff(r_qpose, r_bestmatchpose)   # This takes as an imput quaternions, converts them to rotation vectors, computes the relative rotation between them and returns the absolute angle error            
    
                    t_diff_eplr = t_diff_eplr + diff_t
                    r_diff_eplr = r_diff_eplr + diff_r 
                    
                    t_diff_eplr_list.append(diff_t)
                    r_diff_eplr_list.append(diff_r)
                    
                    gt_bestmatch=int(gt_matches_ep[itr])
                    gt_t_bestmatchpose = self.refimages_ep_poses[gt_bestmatch][0:3]
                    gt_diff_t = np.linalg.norm(t_qpose-gt_t_bestmatchpose)
                    gt_t_diff_eplr_list.append(gt_diff_t)
                    gt_t_diff_eplr_avg=gt_t_diff_eplr_avg+gt_diff_t
                    gt_r_bestmatchpose=self.refimages_ep_poses[gt_bestmatch][3:7]
                    gt_diff_r=self._get_rotation_diff(r_qpose, gt_r_bestmatchpose)
                    gt_r_diff_eplr_avg=gt_r_diff_eplr_avg+gt_diff_r
                    gt_r_diff_eplr_list.append(gt_diff_r)
                                            
                    # self.show_qualitative_matches(itr, match, gt_bestmatch, 'EPLR Office')                

                    
                print('errors computed for LR upsampled DB')        
        
        print('AVG_t_diff_org:',t_diff_org/total_query_images) 
        print('AVG_t_diff_eplr:',t_diff_eplr/total_query_images)
        print('AVG_t_diff_sanitycheck:',t_diff_sanitycheck/total_query_images)
        print('AVG_t_diff_ep:',t_diff_ep/total_query_images)
        

        # print('gt_AVG_t_diff_org:',gt_t_diff_org_avg/total_query_images) 
        # print('gt_AVG_t_diff_ep:',gt_t_diff_ep_avg/total_query_images) 

        print('t_diff_org_median:',np.median(t_diff_org_list)) 
        print('t_diff_eplr_median:',np.median(t_diff_eplr_list)) 
        print('t_diff_sanitycheck_median:',np.median(t_diff_sanitycheck_list)) 
        print('t_diff_ep_median:',np.median(t_diff_ep_list)) 

        print('gt_t_diff_org_median:',np.median(gt_t_diff_org_list)) 
        print('gt_t_diff_ep_median:',np.median(gt_t_diff_ep_list)) 
        
        print('AVG_r_diff_org:',r_diff_org/total_query_images) 
        print('AVG_r_diff_eplr:',r_diff_eplr/total_query_images) 
        print('AVG_r_diff_sanitycheck:',r_diff_sanitycheck/total_query_images) 
        print('AVG_r_diff_ep:',r_diff_ep/total_query_images) 

        # print('gt_AVG_r_diff_org:',gt_r_diff_org_avg/total_query_images) 
        # print('gt_AVG_r_diff_ep:',gt_r_diff_ep_avg/total_query_images) 
        
        print('r_diff_org_median:',np.median(r_diff_org_list)) 
        print('r_diff_eplr_median:',np.median(r_diff_eplr_list))         
        print('r_diff_sanitycheck_median:',np.median(r_diff_sanitycheck_list))         
        print('r_diff_ep_median:',np.median(r_diff_ep_list))         

        print('gt_r_diff_org_median:',np.median(gt_r_diff_org_list)) 
        print('gt_r_diff_ep_median:',np.median(gt_r_diff_ep_list))   
         
        print(np.median(gt_t_diff_org_list)) 
        print(np.median(gt_t_diff_ep_list)) 
        print(np.median(gt_r_diff_org_list)) 
        print(np.median(gt_r_diff_ep_list))   
 
        
        print(t_diff_org/total_query_images) 
        print(t_diff_eplr/total_query_images) 
        print(t_diff_sanitycheck/total_query_images) 
        print(t_diff_ep/total_query_images) 

        print(np.median(t_diff_org_list)) 
        print(np.median(t_diff_eplr_list)) 
        print(np.median(t_diff_sanitycheck_list)) 
        print(np.median(t_diff_ep_list)) 

        
        print(r_diff_org/total_query_images) 
        print(r_diff_eplr/total_query_images) 
        print(r_diff_sanitycheck/total_query_images) 
        print(r_diff_ep/total_query_images) 

        
        print(np.median(r_diff_org_list)) 
        print(np.median(r_diff_eplr_list))   
        print(np.median(r_diff_sanitycheck_list))   
        print(np.median(r_diff_ep_list))   
        
        print('densification_time: ',densification_time)
        print('encoding_time: ', encoding_time)
        print('retrieval_time_sparse: ', retrieval_time_sparse)
        print('retrieval_time_dense: ', retrieval_time_dense)
        print('sparse_map_size: ', sparse_map_size)
        print('dense_map_size: ', dense_map_size)
        
        
        self.visualize_matches(matches_org, matches_ep, matches_eplr, matches_sanitycheck,self.queryimages_poses,self.refimages_ep_poses,self.refimages_poses, gt_matches_org, gt_matches_ep)    
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f'Done')
