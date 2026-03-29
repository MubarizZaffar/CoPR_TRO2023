# What is this code for?
This somewhat unfactored code corresponds to the experiments performed in the CoPR (Zaffar, T-RO 2023) paper. 

There are two stand-alone repositories here: CoPR and COPR_Retrieval_Test_Extrapolation. The first one is used for training the CoPR model and the second one relates to the different types of experiments perfomed in the paper related to CoPR.

# How to use?
All the config parameters are set in the YAML files in /config/ folder. If all dependencies are correctly installed, it could be run with:

    python main.py

This applies for both the respositories.  Some of the models used in the paper are available in: 

> COPR_Retrieval_Test_Extrapolation/models/

The used models for VPR and CoPR were:

**7scenes dataset:** 

> resnet34_originalmodel_ds207scenesdistance_0.0001 and
> resnet34descregressor8linearlayers_resnet34_originalmodel_distanceloss1e4_ds507scenes_do0.0_7scenes_0.000550000
> 
**Station Escalator dataset:** 
> resnet34descregressor8linearlayers_resnet34_originalmodel_station_escalator_v4_station_escalator_0.000550000
> and
> resnet34_originalmodel_station_escalator_v4_station_escalatordistance_5e_05

**Synthetic shopfacade dataset:** 
> resnet34descregressor8linearlayers_resnet34_originalmodel_syntheticshopfacade_train2m25cm_val2m100cm_do0.3synthetic_shopfacadedistance_0.0001_synthetic_shopfacade_0.000550000
> and
> resnet34_originalmodel_syntheticshopfacade_train2m25cm_val2m100cm_do0.3synthetic_shopfacadedistance_0.0001

# Major acknowledgement 
This code is primarily built on top-of the well-written codes of [RelPoseNet](https://github.com/AaltoVision/RelPoseNet). Huge thanks to them for open-sourcing their codes.

# Citation

    @article{zaffar2023copr,
      title={CoPR: Toward accurate visual localization with continuous place-descriptor regression},
      author={Zaffar, Mubariz and Nan, Liangliang and Kooij, Julian Francisco Pieter},
      journal={IEEE Transactions on Robotics},
      volume={39},
      number={4},
      pages={2825--2841},
      year={2023},
      publisher={IEEE}
    }
