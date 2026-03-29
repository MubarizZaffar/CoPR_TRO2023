import random
from os import path as osp
from collections import defaultdict
from PIL import Image
import torch
import pickle
import numpy as np
from scipy.spatial.transform import Rotation as R

class SevenScenesRelPoseRefSingleSequenceDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.cfg = experiment_cfg
        self.chosen_scene = self.cfg.experiment_params.chosen_scene
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)

        if (self.cfg.experiment_params.datasetname=='7scenes'):
            for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='Cambridge'):
            for i, scene in enumerate(['KingsCollege']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene                       
        
        elif (self.cfg.experiment_params.datasetname=='University'):
            for i, scene in enumerate(['office', 'meeting','kitchen1', 'conference','kitchen2']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
            for i, scene in enumerate(['shopfacade']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='relposefailure'):
            for i, scene in enumerate(['office_12scenes', 'sofas_12scenes', 'cabinets_7scenes', 'sofas_7scenes', 'monitors_7scenes', 'topcabinets_7scenes']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            for i, scene in enumerate(['escalator']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
        
        self.data_dir = self.cfg.paths.dataset_path + self.scenes_dict[self.cfg.experiment_params.chosen_scene]
        self.fnames_ref = self._read_images()

    def _get_gt_poses(self): 
        query_poses =  []
        ref_poses =  []

        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/gtposes_test_ds50.txt', 'r') as f:
            for line in f:           
                if (self.cfg.experiment_params.datasetname=='7scenes' or self.cfg.experiment_params.datasetname=='relposefailure'):
                    with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/' + line.rstrip('\n'), 'r') as f:
                        pose = np.empty([7])
                        a = [[float(num) for num in line.split()] for line in f]
                        t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                        r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                        r1_quat=R.from_matrix(r1).as_quat()
                        r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                        pose[0:3] = t1
                        pose[3:8] = r1_quat 
                elif (self.cfg.experiment_params.datasetname=='Cambridge' or self.cfg.experiment_params.datasetname=='University' or self.cfg.experiment_params.datasetname=='synthetic_shopfacade' or self.cfg.experiment_params.datasetname=='station_escalator'):   
                        pose = np.empty([7])
                        line=line.rstrip()
                        linesplit = line.split()
                        for j in range(7):    
                            pose[j] = float(linesplit[j])
                    
                query_poses.append(pose)
        itr=0
        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/gtposes_train_singlesequence.txt', 'r') as f:
            for line in f:
                 if (self.cfg.experiment_params.coprplusrelposeexp==True):   #added for copr+relpose experiment with ds40, otherwise remove this condition
                     if (itr%400==0):
                        if (self.cfg.experiment_params.datasetname=='7scenes' or self.cfg.experiment_params.datasetname=='relposefailure'):
                            with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/' + line.rstrip('\n'), 'r') as f:
                                pose = np.empty([7])
                                a = [[float(num) for num in line.split()] for line in f]
                                t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                                r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                                r1_quat=R.from_matrix(r1).as_quat()
                                r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                                pose[0:3] = t1
                                pose[3:8] = r1_quat 
                                
                        elif (self.cfg.experiment_params.datasetname=='Cambridge' or self.cfg.experiment_params.datasetname=='University' or self.cfg.experiment_params.datasetname=='synthetic_shopfacade' or self.cfg.experiment_params.datasetname=='station_escalator'):   
                                pose = np.empty([7])
                                line=line.rstrip()
                                linesplit = line.split()
                                for j in range(7):    
                                    pose[j] = float(linesplit[j])
                                    
                        ref_poses.append(pose)
                                    
                 else:   
                        if (self.cfg.experiment_params.datasetname=='7scenes' or self.cfg.experiment_params.datasetname=='relposefailure'):
                            with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/' + line.rstrip('\n'), 'r') as f:
                                pose = np.empty([7])
                                a = [[float(num) for num in line.split()] for line in f]
                                t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                                r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                                r1_quat=R.from_matrix(r1).as_quat()
                                r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                                pose[0:3] = t1
                                pose[3:8] = r1_quat 
                                
                        elif (self.cfg.experiment_params.datasetname=='Cambridge' or self.cfg.experiment_params.datasetname=='University' or self.cfg.experiment_params.datasetname=='synthetic_shopfacade' or self.cfg.experiment_params.datasetname=='station_escalator'):   
                                pose = np.empty([7])
                                line=line.rstrip()
                                linesplit = line.split()
                                for j in range(7):    
                                    pose[j] = float(linesplit[j])                                
                            
                        ref_poses.append(pose)
            
                 itr=itr+1
            print('length of rposes is: ',len(ref_poses))
                                                        
        return query_poses, ref_poses
                     
    def _read_images(self):        
        fnames = []
        itr=0
        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/filenames_train_singlesequence.txt', 'r') as f:
            for line in f:
                if (self.cfg.experiment_params.coprplusrelposeexp==True):
                    if (itr%400==0): #added for copr+relpose experiment with ds40, otherwise remove this condition
                        chunks = line.rstrip()
                        fnames.append(chunks)
                        
                else:
                    chunks = line.rstrip()
                    fnames.append(chunks)
                    
                itr=itr+1
            
        return fnames

    def __getitem__(self, item):
        img = Image.open(self.data_dir+'/'+self.fnames_ref[item]).convert('RGB')

        if self.transforms:
            img = self.transforms(img)

        return {'img': img, 'index':item
                }

    def __len__(self):
        return len(self.fnames_ref)
    
class SevenScenesRelPoseRefExtrapolatedDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.cfg = experiment_cfg
        self.chosen_scene = self.cfg.experiment_params.chosen_scene
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        if (self.cfg.experiment_params.datasetname=='7scenes'):
            for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='Cambridge'):
            for i, scene in enumerate(['KingsCollege']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene   
        
        elif (self.cfg.experiment_params.datasetname=='University'):
            for i, scene in enumerate(['office', 'meeting','kitchen1', 'conference','kitchen2']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
            for i, scene in enumerate(['shopfacade']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='relposefailure'):
            for i, scene in enumerate(['office_12scenes', 'sofas_12scenes', 'cabinets_7scenes', 'sofas_7scenes', 'monitors_7scenes', 'topcabinets_7scenes']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            for i, scene in enumerate(['escalator']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        self.data_dir = self.cfg.paths.dataset_path + self.scenes_dict[self.cfg.experiment_params.chosen_scene]
        
        self.anchornames, self.regressionnodesnames, self.rel_poses= self._read_images()

    def _read_images(self):        
        anchornames = []
        regressionnodesnames = []

        rel_poses = []
        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(self.cfg.experiment_params.x_upsampling)+'_y'+str(self.cfg.experiment_params.y_upsampling)+'_ss'+str(self.cfg.experiment_params.stepsize)+'.txt', 'r') as f:
            for line in f:
                chunks = line.rstrip().split()
                
                if (self.cfg.experiment_params.coprplusrelposeexp==True):
                    if(chunks[0]!=chunks[1]): #That is only add regressed descriptors and not the dense anchor points on ref trajectory
                        anchornames.append(chunks[0])
                        regressionnodesnames.append(chunks[1])
                        rel_poses.append(torch.as_tensor([float(chunks[2]),float(chunks[3]) \
                                                          ,float(chunks[4]),float(chunks[5]) \
                                                          ,float(chunks[6]),float(chunks[7])  \
                                                          ,float(chunks[8])]))
                else:
                    anchornames.append(chunks[0])
                    regressionnodesnames.append(chunks[1])
                    rel_poses.append(torch.as_tensor([float(chunks[2]),float(chunks[3]) \
                                                      ,float(chunks[4]),float(chunks[5]) \
                                                      ,float(chunks[6]),float(chunks[7])  \
                                                      ,float(chunks[8])]))            
        return anchornames, regressionnodesnames, rel_poses

    def _get_gt_poses(self): 
        anchornames_all = [] #these _all arrays are used to select gt poses for extrapolatopm experiments with copr+relpose since I do not intend to use the dense reference points but only sparse ones
        regressionnodesnames_all = []
        if (self.cfg.experiment_params.coprplusrelposeexp==True):
            with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(self.cfg.experiment_params.x_upsampling)+'_y'+str(self.cfg.experiment_params.y_upsampling)+'_ss'+str(self.cfg.experiment_params.stepsize)+'.txt', 'r') as f:
                for line in f:
                    chunks = line.rstrip().split()
                    anchornames_all.append(chunks[0])
                    regressionnodesnames_all.append(chunks[1])


        ref_poses_extrapolated =  []

        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/gtposes_train_extrapolated'+'_x'+str(self.cfg.experiment_params.x_upsampling)+'_y'+str(self.cfg.experiment_params.y_upsampling)+'_ss'+str(self.cfg.experiment_params.stepsize)+'.txt', 'r') as f:
            i=0
            for line in f:
                if (self.cfg.experiment_params.coprplusrelposeexp==True):
                    if (anchornames_all[i]!=regressionnodesnames_all[i]):
                        pose = np.empty([7])
                        line=line.rstrip()
                        linesplit = line.split()
                        for itr in range(7):    
                            pose[itr] = float(linesplit[itr])
                            
                        ref_poses_extrapolated.append(pose)

                else:
                    if (self.cfg.experiment_params.datasetname=='relposefailure'):
                        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/' + line.rstrip('\n'), 'r') as f:
                            pose = np.empty([7])
                            a = [[float(num) for num in line.split()] for line in f]
                            t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                            r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                            r1_quat=R.from_matrix(r1).as_quat()
                            r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                            pose[0:3] = t1
                            pose[3:8] = r1_quat 
                            
                    elif (self.cfg.experiment_params.datasetname=='7scenes' or self.cfg.experiment_params.datasetname=='Cambridge' or self.cfg.experiment_params.datasetname=='University' or self.cfg.experiment_params.datasetname=='synthetic_shopfacade' or self.cfg.experiment_params.datasetname=='station_escalator'):   
                            pose = np.empty([7])
                            line=line.rstrip()
                            linesplit = line.split()
                            for j in range(7):    
                                pose[j] = float(linesplit[j]) 
                        
                    ref_poses_extrapolated.append(pose)  
                    
                i=i+1                       
        return ref_poses_extrapolated
    
    def __getitem__(self, item):
        img = Image.open(self.data_dir+'/'+self.anchornames[item]).convert('RGB')
        
        if (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):    
            regimg = Image.open(self.data_dir+'/'+self.regressionnodesnames[item]).convert('RGB') 

        else:
            regimg = img
            
        rel_pose = self.rel_poses[item]
        requires_regression = True
        if self.anchornames[item]==self.regressionnodesnames[item] : requires_regression=False
        
        if self.transforms:
            img = self.transforms(img)
            regimg = self.transforms(regimg)

        return {'img': img, 'regimg': regimg,'index':item, 'rel_pose':rel_pose, 'requires_regression':requires_regression
                }

    def __len__(self):
        return len(self.anchornames)

class SevenScenesRelPoseRefEPLRDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.cfg = experiment_cfg
        self.chosen_scene = self.cfg.experiment_params.chosen_scene
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        if (self.cfg.experiment_params.datasetname=='7scenes'):
            for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='Cambridge'):
            for i, scene in enumerate(['KingsCollege']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene   
        
        elif (self.cfg.experiment_params.datasetname=='University'):
            for i, scene in enumerate(['office', 'meeting','kitchen1', 'conference','kitchen2']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
            for i, scene in enumerate(['shopfacade']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='relposefailure'):
            for i, scene in enumerate(['office_12scenes', 'sofas_12scenes', 'cabinets_7scenes', 'sofas_7scenes', 'monitors_7scenes', 'topcabinets_7scenes']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            for i, scene in enumerate(['escalator']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        self.data_dir = self.cfg.paths.dataset_path + self.scenes_dict[self.cfg.experiment_params.chosen_scene]
        
        self.anchor1names, self.anchor2names, self.anchor3names, self. anchor4names, self.regressionnodesnames, \
        self.tposes1, self.tposes2, self.tposes3, self.tposes4, self.tposesreg = self._read_images()

    def _read_images(self):        
        anchor1names = []
        anchor2names = []
        anchor3names = []
        anchor4names = []
        
        regressionnodesnames = []
        tposes1 = []
        tposes2 = []
        tposes3 = []
        tposes4 = []
        
        tposesreg = []
        
        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(self.cfg.experiment_params.x_upsampling)+'_y'+str(self.cfg.experiment_params.y_upsampling)+'_ss'+str(self.cfg.experiment_params.stepsize)+'.txt', 'r') as f:
            for line in f:
                chunks = line.rstrip().split()
                anchor1names.append(chunks[0])
                anchor2names.append(chunks[1])
                anchor3names.append(chunks[2])                
                anchor4names.append(chunks[3])                

                regressionnodesnames.append(chunks[4])
                
                if (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
                    # since this dataset has only 2D (x, y) variations
                    tposes1.append(np.asarray([float(chunks[5]),float(chunks[6]) \
                                                      ]))
                    tposes2.append(np.asarray([float(chunks[8]),float(chunks[9]) \
                                                      ]))
                    tposes3.append(np.asarray([float(chunks[11]),float(chunks[12]) \
                                                      ]))
                    tposes4.append(np.asarray([float(chunks[14]),float(chunks[15]) \
                                                      ]))
                    
                    tposesreg.append(np.asarray([float(chunks[17]),float(chunks[18]) \
                                                      ]))
                else:
                    tposes1.append(np.asarray([float(chunks[5]),float(chunks[6]) \
                                                      ,float(chunks[7])]))
                    tposes2.append(np.asarray([float(chunks[8]),float(chunks[9]) \
                                                      ,float(chunks[10])]))
                    tposes3.append(np.asarray([float(chunks[11]),float(chunks[12]) \
                                                      ,float(chunks[13])]))
                    tposes4.append(np.asarray([float(chunks[14]),float(chunks[15]) \
                                                      ,float(chunks[16])]))
                    
                    tposesreg.append(np.asarray([float(chunks[17]),float(chunks[18]) \
                                                      ,float(chunks[19])]))

            
        return anchor1names, anchor2names, anchor3names, anchor4names, regressionnodesnames, tposes1, tposes2, tposes3, tposes4, tposesreg

    def _get_gt_poses(self): 
        ref_poses_extrapolated =  []

        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/gtposes_train_extrapolated'+'_x'+str(self.cfg.experiment_params.x_upsampling)+'_y'+str(self.cfg.experiment_params.y_upsampling)+'_ss'+str(self.cfg.experiment_params.stepsize)+'.txt', 'r') as f:
            for line in f:
                pose = np.empty([7])
                line=line.rstrip()
                linesplit = line.split()
                for itr in range(7):    
                    pose[itr] = float(linesplit[itr])
                    
                ref_poses_extrapolated.append(pose)
                                           
        return ref_poses_extrapolated
    
    def __getitem__(self, item):
        anchor1img = Image.open(self.data_dir+'/'+self.anchor1names[item]).convert('RGB')
        anchor2img = Image.open(self.data_dir+'/'+self.anchor2names[item]).convert('RGB')
        anchor3img = Image.open(self.data_dir+'/'+self.anchor3names[item]).convert('RGB')
        anchor4img = Image.open(self.data_dir+'/'+self.anchor4names[item]).convert('RGB')
        
        tpose1 = self.tposes1[item]
        tpose2 = self.tposes2[item]
        tpose3 = self.tposes3[item]
        tpose4 = self.tposes4[item]
        tposereg = self.tposesreg[item]
        requires_regression = True
        
        if self.anchor1names[item]==self.regressionnodesnames[item] : requires_regression=False
        
        if self.transforms:
            anchor1img = self.transforms(anchor1img)
            anchor2img = self.transforms(anchor2img)
            anchor3img = self.transforms(anchor3img)
            anchor4img = self.transforms(anchor4img)

        return {'anchor1img': anchor1img, 'anchor2img': anchor2img, 'anchor3img': anchor3img, 'anchor4img': anchor4img, 
                'index':item, 'requires_regression':requires_regression, 'tpose1': tpose1,'tpose2':tpose2, 
                'tpose3':tpose3, 'tpose4':tpose4, 'tposereg':tposereg
                }

    def __len__(self):
        return len(self.anchor1names)
    
class SevenScenesRelPoseQueryDataset(object):
    def __init__(self, experiment_cfg, transforms=None):
        self.cfg = experiment_cfg
        self.transforms = transforms
        self.scenes_dict = defaultdict(str)
        if (self.cfg.experiment_params.datasetname=='7scenes'):
            for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='Cambridge'):
            for i, scene in enumerate(['KingsCollege']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene   

        elif (self.cfg.experiment_params.datasetname=='University'):
            for i, scene in enumerate(['office', 'meeting','kitchen1', 'conference','kitchen2']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='synthetic_shopfacade'):
            for i, scene in enumerate(['shopfacade']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene

        elif (self.cfg.experiment_params.datasetname=='relposefailure'):
            for i, scene in enumerate(['office_12scenes', 'sofas_12scenes', 'cabinets_7scenes', 'sofas_7scenes', 'monitors_7scenes', 'topcabinets_7scenes']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                
        elif (self.cfg.experiment_params.datasetname=='station_escalator'):
            for i, scene in enumerate(['escalator']): #for potentially other scenes as well 
                self.scenes_dict[i] = scene
                        
        self.data_dir = self.cfg.paths.dataset_path + self.scenes_dict[self.cfg.experiment_params.chosen_scene]
        
        self.fnames_query = self._read_images()

    def _read_images(self):
        qnames = []
       
        with open(self.cfg.paths.dataset_path + '/' + self.scenes_dict[self.cfg.experiment_params.chosen_scene] + '/filenames_test_ds50.txt', 'r') as f:
            for line in f:
                chunks = line.rstrip()
                qnames.append(chunks)
             
        return qnames

    def __getitem__(self, item):
        img = Image.open(self.data_dir+'/'+self.fnames_query[item]).convert('RGB')
      
        if self.transforms:
            img = self.transforms(img)
      
        return {'img': img, 'index':item}

    def __len__(self):
        return len(self.fnames_query)