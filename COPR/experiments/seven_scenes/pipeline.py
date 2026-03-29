import os
from os import path as osp
from tqdm import tqdm
import torch
from service.benchmark_base import Benchmark
from dataset import SevenScenesRelPoseDataset
from augmentations import get_augmentations
from model import RelPoseNet
import pickle

class SevenScenesBenchmark(Benchmark):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.dataloader = self._init_dataloader()
        self.model = self._load_model_relposenet().to(self.device)

    def _init_dataloader(self):
        experiment_cfg = self.cfg.train_params#self.cfg.experiment.experiment_params

        # define test augmentations
        train_augs, eval_aug = get_augmentations()

        # test dataset
        dataset = SevenScenesRelPoseDataset(self.cfg, split='train', transforms=eval_aug)

        # define a dataloader
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=experiment_cfg.bs,
                                                 shuffle=False,
                                                 pin_memory=True,
                                                 num_workers=experiment_cfg.n_workers,
                                                 drop_last=False)

        return dataloader
    
    def _load_model_relposenet(self):
        print(f'Loading RelPoseNet model...')
        model_params_cfg = self.cfg
        
        modelrelposenetorg = RelPoseNet(model_params_cfg)
        
        if (self.cfg.data_params.loss_type == 'relativepose'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenetorg.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with relative pose... Done!')

        elif (self.cfg.data_params.loss_type == 'triplet'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenettriplet.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with triplet... Done!')

        elif (self.cfg.data_params.loss_type == 'distance'):
            data_dict = torch.load(model_params_cfg.model_paramsrelposenetdistancebased.snapshot)
            modelrelposenetorg.load_state_dict(data_dict['state_dict'])
            print(f'Loading RelPoseNet model trained with distance based... Done!')
            
        return modelrelposenetorg.eval()

    def evaluate(self):
        q_est_all, t_est_all = [], []
        desc_dict = {}
        print(f'Evaluate on the dataset...')
        with torch.no_grad():
            for data_batch in tqdm(self.dataloader):
                feat1, feat2, q_est, t_est = self.model(data_batch['img1'].to(self.device),
                                          data_batch['img2'].to(self.device))
                # print(data_batch['img1name'][0])
                # print(data_batch['img2name'][0])
                desc_dict[data_batch['img1name'][0]]=feat1.detach().cpu().numpy()
                desc_dict[data_batch['img2name'][0]]=feat2.detach().cpu().numpy()
                q_est_all.append(q_est)
                t_est_all.append(t_est)

        output="train_descs_resnet34_originalmodel_ds507scenes_5e_05_stationescalator.pkl" # train_descs_resnet34_originalmodel_distanceloss1e4_ds507scenes or val_descs_resnet34_originalmodel_ds507scenes_5e_05_shopfacade_2m100cm_top20
        descdict_file = open(output,"wb")  #val_descs_resnet34_originalmodelmixed_reftrajectasTestset.pkl
        print(output)
        desc_dict = pickle.dump(desc_dict,descdict_file)
        descdict_file.close()
        
        # q_est_all = torch.cat(q_est_all).cpu().numpy()
        # t_est_all = torch.cat(t_est_all).cpu().numpy()

        # print(f'Write the estimates to a text file')
        # experiment_cfg = self.cfg.experiment.experiment_params

        # if not osp.exists(experiment_cfg.output.home_dir):
        #     os.makedirs(experiment_cfg.output.home_dir)

        # with open(experiment_cfg.output.res_txt_fname, 'w') as f:
        #     for q_est, t_est in zip(q_est_all, t_est_all):
        #         f.write(f"{q_est[0]} {q_est[1]} {q_est[2]} {q_est[3]} {t_est[0]} {t_est[1]} {t_est[2]}\n")

        print(f'Done')
