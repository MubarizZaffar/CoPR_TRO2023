#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 14:33:41 2021

@author: mzaffar
"""

from scipy.spatial.transform import Rotation as R
import numpy as np
dataset='7scenes'

if (dataset=='7scenes'):
    selected_scenes=np.arange(0,7)
    scenes_dict={}

    selected_scene=4
    sequence_to_use=0
    stepsize=0.35
    x_upsampling=1.6
    y_upsampling=1.0

    training_sequences=[]    
    itr=0
    
    for i, scene in enumerate(['chess', 'fire', 'heads', 'office', 'pumpkin', 'redkitchen', 'stairs']):
        scenes_dict[i] = scene
    
        
    def get_rotation_diff(r_bestmatchpose, r_qpose):
        r1 = R.from_quat(r_bestmatchpose)
        r2 = R.from_quat(r_qpose)
        
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        rot_diff = np.matmul(np.transpose(r1), r2)
        rot_diff = R.from_matrix(rot_diff).as_quat()
        rot_diff=[rot_diff[3],rot_diff[0],rot_diff[1],rot_diff[2]]
        
        return rot_diff
        
    
    poses=[]
    names=[]
    scene_ids=[]
    
    #automating which sequence is to be used as the base sequence for upsampling
    with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+ '/TrainSplit.txt', 'r') as f:
        for line in f:
            line=line.rstrip()
            seq=int(line[-1])
            print(line[-1])
            
            training_sequences.append('seq-0'+str(seq))
                    
    with open ('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/filenames_train_ds20.txt') as f:
        for line in f:
            if (training_sequences[sequence_to_use] in line):
                with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/filenames_train_singlesequence.txt', 'a') as the_file:
                    print('')
                    # the_file.write(line)
    
    with open ('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_ds20.txt') as f:
        for line in f:
            if (training_sequences[sequence_to_use] in line):
                with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt', 'a') as the_file:
                    print('')
                    # the_file.write(line)
                    
    with open ('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt') as f: # creating the list of names and poses to be used in following loop
        for line in f:
            with open('/home/mzaffar/Documents/datasets/7scenes/' + scenes_dict[selected_scene] + '/' + line.rstrip('\n'), 'r') as f:
                    pose = np.empty([7])
                    a = [[float(num) for num in line.split()] for line in f]
                    t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                    r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                    r1_quat=R.from_matrix(r1).as_quat()
                    r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                    pose[0:3] = t1
                    pose[3:7] = r1_quat
                    poses.append(pose)
                    names.append(line.rstrip('\n'))                    
    
    with open ('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt') as f:
        for line in f:
            with open('/home/mzaffar/Documents/datasets/7scenes/' + scenes_dict[selected_scene] + '/' + line.rstrip('\n'), 'r') as f:
                    pose = np.empty([7])
                    a = [[float(num) for num in line.split()] for line in f]
                    t1=np.asarray([a[0][3],a[1][3],a[2][3]])
                    r1 = np.asarray([a[0][0:3], a[1][0:3],a[2][0:3]])
                    r1_quat=R.from_matrix(r1).as_quat()
                    r1_quat=[r1_quat[3],r1_quat[0],r1_quat[1],r1_quat[2]]
                    pose[0:3] = t1
                    pose[3:7] = r1_quat
                    # poses.append(pose)
                    # names.append(line.rstrip('\n'))
                    name=line.rstrip('\n')
                    scene_ids.append(selected_scene)
    
                    with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                        posestr = [float("{:.5f}".format(a)) for a in pose]
                        the_file.write(str(posestr[0]) + ' '  + str(posestr[1]) + ' '  + str(posestr[2]) + ' '  + str(posestr[3]) + ' '  + str(posestr[4]) + ' '  + str(posestr[5]) + ' '  + str(posestr[6])+'\n')                
    
                    with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                        imgname=name.replace(".pose.txt",".color.png")
                        output='/'+ imgname + ' ' + '/' + imgname + ' ' + str(0) + ' '  + str(0) + ' '  + str(0) + ' '  + str(1) + ' '  + str(0) + ' '  + str(0) + ' '  + str(0)
                        # print(output)
                        the_file.write(output+'\n')

                    with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                        imgname=name.replace(".pose.txt",".color.png")
                        output='/'+ imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) \
                           + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0)
                        # print(output)
                        the_file.write(output+'\n')    
                    
                    if (itr%40==0):
                        anchor_name=name.replace(".pose.txt",".color.png")
                        anchor_pose=pose.copy()
                        
                        # far anchors for EPLR
                        # if (itr-200 < 0):
                        #     anchor1_index=itr
                        #     anchor2_index=itr+100
                        #     anchor3_index=itr+200
                        #     anchor4_index=itr+300
                        #     print('here1') 


                        # elif (itr+100 > len(names)-1):
                        #     anchor1_index=itr-300
                        #     anchor2_index=itr-200
                        #     anchor3_index=itr-100
                        #     anchor4_index=itr
                            
                            
                            
                        #     print('here2') 

                        # else:
                        #     anchor1_index=itr-200
                        #     anchor2_index=itr-100
                        #     anchor3_index=itr
                        #     anchor4_index=itr+100
                        #     print('here3') 
                        
                        # Near anchors for EPLR
                        if (itr-40 < 0):
                            anchor1_index=itr
                            anchor2_index=itr+40
                            anchor3_index=itr+80
                            anchor4_index=itr+120
                            # print('here1') 


                        elif (itr+80 > len(names)-1):
                            if (itr+40 < len(names)-1):
                                anchor1_index=itr-80
                                anchor2_index=itr-40
                                anchor3_index=itr
                                anchor4_index=itr+40-1
                            else:
                                anchor1_index=itr-80
                                anchor2_index=itr-40
                                anchor3_index=itr
                                anchor4_index=itr+20-1
                            
                            
                            
                            # print('here2') 

                        else:
                            anchor1_index=itr-40
                            anchor2_index=itr
                            anchor3_index=itr+40
                            anchor4_index=itr+80
                            # print('here3') 
                            


                        for x in np.arange(-1*x_upsampling/2,x_upsampling/2,stepsize):
                            for y in np.arange(-1*y_upsampling/2,y_upsampling/2,stepsize):
                                posenew=anchor_pose.copy()
                                # print(posenew)
                                # print(x)
                                # print(y)
                                posenew[0]=posenew[0]+x
                                posenew[1]=posenew[1]+y
                                # print(posenew)
    
                                with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                                    posestrnew = [float("{:.5f}".format(a)) for a in posenew]
                                    out2=str(posestrnew[0]) + ' '  + str(posestrnew[1]) + ' '  + str(posestrnew[2]) + ' '  + str(posestrnew[3]) + ' '  + str(posestrnew[4]) + ' '  + str(posestrnew[5]) + ' '  + str(posestrnew[6])+'\n'
                                    # print(out2)
                                    the_file.write(out2)                
                                   
                                namenew=line.rstrip('\n').split('.')[0]
                                namenew=namenew+str("{:.3f}".format(x))+str("{:.3f}".format(y))+".color.png"
    
                                r_pose_t = np.asarray(posenew[0:3])-np.asarray(anchor_pose[0:3])
                                r_pose_quat = get_rotation_diff(anchor_pose[3:7], posenew[3:7]) 
                                r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                                r_pose = [float("{:.5f}".format(a)) for a in r_pose]
                                
                                with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                                    output='/'+anchor_name + ' ' + '/'+namenew +' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) +'\n'
                                    the_file.write(output)
                                    # print(output)

                                print(itr)
                                print(anchor1_index)
                                print(anchor2_index)
                                print(anchor3_index)
                                print(anchor4_index)
                                                     
                                anchor1name=names[anchor1_index].replace(".pose.txt",".color.png")
                                anchor2name=names[anchor2_index].replace(".pose.txt",".color.png")
                                anchor3name=names[anchor3_index].replace(".pose.txt",".color.png")
                                anchor4name=names[anchor4_index].replace(".pose.txt",".color.png")
           
                                t1_pose=poses[anchor1_index][0:3]
                                print(t1_pose)
                                t1_pose = [float("{:.5f}".format(a)) for a in t1_pose]    
                                print(t1_pose)

                                t2_pose=poses[anchor2_index][0:3]                                
                                t2_pose = [float("{:.5f}".format(a)) for a in t2_pose]    
                                
                                t3_pose=poses[anchor3_index][0:3]                                
                                t3_pose = [float("{:.5f}".format(a)) for a in t3_pose]    
                                
                                t4_pose=poses[anchor4_index][0:3]                                
                                t4_pose = [float("{:.5f}".format(a)) for a in t4_pose]    
                                
                                treg_pose=posenew[0:3]                                
                                treg_pose = [float("{:.5f}".format(a)) for a in treg_pose]
                                
                                with open('/home/mzaffar/Documents/datasets/7scenes/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(x_upsampling)+'_y'+str(y_upsampling)+'_ss'+str(stepsize)+'.txt', 'a') as the_file:
                                    output='/' + anchor1name + ' ' + '/' + anchor2name + ' ' + '/' + anchor3name + ' ' + '/' + anchor4name + ' ' + '/' + namenew + ' ' \
                                    + str(t1_pose[0]) + ' '  + str(t1_pose[1]) + ' '  + str(t1_pose[2]) + ' '  \
                                    + str(t2_pose[0]) + ' '  + str(t2_pose[1]) + ' '  + str(t2_pose[2]) + ' ' \
                                    + str(t3_pose[0]) + ' '  + str(t3_pose[1]) + ' '  + str(t3_pose[2]) + ' ' \
                                    + str(t4_pose[0]) + ' '  + str(t4_pose[1]) + ' '  + str(t4_pose[2]) + ' ' \
                                    + str(treg_pose[0]) + ' '  + str(treg_pose[1]) + ' '  + str(treg_pose[2]) + ' ' + '\n'
                                    the_file.write(output)
                                    # print(output)
    
    
                    itr=itr+1
    print("The sequence used for upsampling was: ", training_sequences[sequence_to_use] )
    
elif (dataset=='cambridge'):

    selected_scenes=np.arange(0,1)
    scenes_dict={}
    selected_scene=0
    
    stepsize=1
    x_upsampling=8
    y_upsampling=20
    sequence_to_use=0
    anchor_samplingrate=20
    training_sequences=[]
    
    itr=0
    
    for i, scene in enumerate(['KingsCollege']): #for potentially other scenes as well 
        scenes_dict[i] = scene    
        
    def get_rotation_diff(r_bestmatchpose, r_qpose):
        r1 = R.from_quat(r_bestmatchpose)
        r2 = R.from_quat(r_qpose)
        
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        rot_diff = np.matmul(np.transpose(r1), r2)
        rot_diff = R.from_matrix(rot_diff).as_quat()
        rot_diff=[rot_diff[3],rot_diff[0],rot_diff[1],rot_diff[2]]
        
        return rot_diff
        
    
    poses=[]
    names=[]
    scene_ids=[]
    
    #automating which sequence is to be used as the base sequence for upsampling
    with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+ '/TrainSplit.txt', 'r') as f:
        for line in f:
            line=line.rstrip()
            seq=int(line[-1])
            print(line[-1])
            if (seq==1):         
                training_sequences.append("seq1")
            if (seq==2):            
                training_sequences.append("seq2")
            if (seq==3):            
                training_sequences.append("seq3")
            if (seq==4):            
                training_sequences.append("seq4")
            if (seq==5):            
                training_sequences.append("seq5")
            if (seq==6):            
                training_sequences.append("seq6")
            if (seq==7):            
                training_sequences.append("seq7")
            if (seq==8):            
                training_sequences.append("seq8")
            if (seq==9):            
                training_sequences.append("seq9")
                
    poses=[]
    names_ss=[]
    
    with open ('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train.txt') as f:
        for line in f:
            poses.append(line)
                    
    with open ('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/filenames_train.txt') as f:
        for it,line in enumerate(f):
            if (training_sequences[sequence_to_use] in line):
                with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/filenames_train_singlesequence.txt', 'a') as the_file:
                    the_file.write(line)
                    names_ss.append(line)
                with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt', 'a') as the_file:
                    the_file.write(poses[it])
    
    # with open ('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train.txt') as f:
    #     for line in f:
    #         if (training_sequences[sequence_to_use] in line):
    #             with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt', 'a') as the_file:
    #                 the_file.write(line)
    
    with open ('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt') as f:
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            # poses.append(pose)
            # names.append(line.rstrip('\n'))
            name=names_ss[itr].rstrip('\n')
            scene_ids.append(selected_scene)
    
            with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated.txt', 'a') as the_file:
                posestr = [float("{:.5f}".format(a)) for a in pose]
                the_file.write(str(posestr[0]) + ' '  + str(posestr[1]) + ' '  + str(posestr[2]) + ' '  + str(posestr[3]) + ' '  + str(posestr[4]) + ' '  + str(posestr[5]) + ' '  + str(posestr[6])+'\n')                

            with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'.txt', 'a') as the_file:
                imgname=name.replace(".pose.txt",".color.png")
                output='/'+ imgname + ' ' + '/' + imgname + ' ' + str(0) + ' '  + str(0) + ' '  + str(0) + ' '  + str(1) + ' '  + str(0) + ' '  + str(0) + ' '  + str(0)
                the_file.write(output+'\n')

            
            if (itr%anchor_samplingrate==0):
                anchor_name=name
                anchor_pose=pose.copy()
                for x in np.arange(-1*x_upsampling/2,x_upsampling/2,stepsize):
                    for y in np.arange(-1*y_upsampling/2,y_upsampling/2,stepsize):
                        posenew=anchor_pose.copy()
                        print(posenew)
                        print(x)
                        print(y)
                        posenew[0]=posenew[0]+x
                        posenew[1]=posenew[1]+y
                        print(posenew)

                        with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated.txt', 'a') as the_file:
                            posestrnew = [float("{:.5f}".format(a)) for a in posenew]
                            the_file.write(str(posestrnew[0]) + ' '  + str(posestrnew[1]) + ' '  + str(posestrnew[2]) + ' '  + str(posestrnew[3]) + ' '  + str(posestrnew[4]) + ' '  + str(posestrnew[5]) + ' '  + str(posestrnew[6])+'\n')                
                           
                        # namenew=line.rstrip('\n').split('.')[0]
                        namenew=name.replace('.png','')+str("{:.3f}".format(x))+str("{:.3f}".format(y))+".png"

                        r_pose_t = np.asarray(posenew[0:3])-np.asarray(anchor_pose[0:3])
                        r_pose_quat = get_rotation_diff(anchor_pose[3:7], posenew[3:7]) 
                        r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                        r_pose = [float("{:.5f}".format(a)) for a in r_pose]
                        
                        with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'.txt', 'a') as the_file:
                            output='/'+anchor_name + ' ' + '/'+namenew +' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6])
                            the_file.write(output+'\n')
                        

            itr=itr+1
    print("The sequence used for upsampling was: ", training_sequences[sequence_to_use] )
    
elif (dataset=='University'):
    selected_scenes=np.arange(0,5)
    scenes_dict={}
    selected_scene=4
    
    stepsize=0.5
    x_upsampling=1
    y_upsampling=3.5
    sequence_to_use=0
    anchor_samplingrate=20
    training_sequences=[]
    
    itr=0
    
    for i, scene in enumerate(['office', 'meeting','kitchen1', 'conference','kitchen2']): #for potentially other scenes as well 
        scenes_dict[i] = scene    
        
    def get_rotation_diff(r_bestmatchpose, r_qpose):
        r1 = R.from_quat(r_bestmatchpose)
        r2 = R.from_quat(r_qpose)
        
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        rot_diff = np.matmul(np.transpose(r1), r2)
        rot_diff = R.from_matrix(rot_diff).as_quat()
        rot_diff=[rot_diff[3],rot_diff[0],rot_diff[1],rot_diff[2]]
        
        return rot_diff
        
    
    poses=[]
    names=[]
    scene_ids=[]
    
    #automating which sequence is to be used as the base sequence for upsampling
    with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+ '/TrainSplit.txt', 'r') as f:
        for line in f:
            line=line.rstrip()
            seq=int(line[-1])
            print(line[-1])
            training_sequences.append("seq_0"+str(seq))

                
    poses=[]
    names_ss=[]
    poses_ss=[]
    
    with open ('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/gtposes_train_ds20.txt') as f:
        for line in f:
            poses.append(line)
                    
    with open ('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/filenames_train_ds20.txt') as f:
        for it,line in enumerate(f):
            if (training_sequences[sequence_to_use] in line):
                with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/filenames_train_singlesequence.txt', 'a') as the_file:
                    # the_file.write(line)
                    names_ss.append(line.rstrip('\n'))
                with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt', 'a') as the_file:
                    pose = np.empty([7])
                    line=poses[it].rstrip()
                    linesplit = line.split()
                    for j in range(7):    
                        pose[j] = float(linesplit[j])
                    poses_ss.append(pose)
                    # the_file.write(poses[it])
    
    # with open ('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train.txt') as f:
    #     for line in f:
    #         if (training_sequences[sequence_to_use] in line):
    #             with open('/home/mzaffar/Documents/datasets/Cambridge/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt', 'a') as the_file:
    #                 the_file.write(line)
    
    with open ('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/gtposes_train_singlesequence.txt') as f:
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            # poses.append(pose)
            # names.append(line.rstrip('\n'))
            name=names_ss[itr].rstrip('\n')
            scene_ids.append(selected_scene)
    
            with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated.txt', 'a') as the_file:
                posestr = [float("{:.5f}".format(a)) for a in pose]
                # the_file.write(str(posestr[0]) + ' '  + str(posestr[1]) + ' '  + str(posestr[2]) + ' '  + str(posestr[3]) + ' '  + str(posestr[4]) + ' '  + str(posestr[5]) + ' '  + str(posestr[6])+'\n')                

            with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'.txt', 'a') as the_file:
                imgname=name.replace(".pose.txt",".color.png")
                output='/'+ imgname + ' ' + '/' + imgname + ' ' + str(0) + ' '  + str(0) + ' '  + str(0) + ' '  + str(1) + ' '  + str(0) + ' '  + str(0) + ' '  + str(0)
                # the_file.write(output+'\n')

            with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_eplr'+'.txt', 'a') as the_file:
                imgname=name.replace(".pose.txt",".color.png")
                output='/'+ imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + '/' + imgname + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) \
                           + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0)

                the_file.write(output+'\n')
            
            if (itr%anchor_samplingrate==0):
                anchor_name=name
                anchor_pose=pose.copy()
                
                if (itr-anchor_samplingrate < 0):
                    anchor1_index=itr
                    anchor2_index=itr+anchor_samplingrate
                    anchor3_index=itr+2*anchor_samplingrate
                    anchor4_index=itr+3*anchor_samplingrate
                    
                elif (itr+2*anchor_samplingrate > len(names_ss)-1):
                    if (itr+anchor_samplingrate < len(names_ss)-1):
                        anchor1_index=itr-2*anchor_samplingrate
                        anchor2_index=itr-anchor_samplingrate
                        anchor3_index=itr
                        anchor4_index=itr+anchor_samplingrate-1
                    else:
                        anchor1_index=itr-2*anchor_samplingrate
                        anchor2_index=itr-anchor_samplingrate
                        anchor3_index=itr
                        anchor4_index=itr+int(anchor_samplingrate/2)-1


                else:
                    anchor1_index=itr-anchor_samplingrate
                    anchor2_index=itr
                    anchor3_index=itr+anchor_samplingrate
                    anchor4_index=itr+2*anchor_samplingrate
                            
                for x in np.arange(-1*x_upsampling/2,x_upsampling/2,stepsize):
                    for y in np.arange(-1*y_upsampling/2,y_upsampling/2,stepsize):
                        posenew=anchor_pose.copy()
                        print(posenew)
                        print(x)
                        print(y)
                        posenew[0]=posenew[0]+x
                        posenew[1]=posenew[1]+y
                        print(posenew)

                        with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/gtposes_train_extrapolated.txt', 'a') as the_file:
                            posestrnew = [float("{:.5f}".format(a)) for a in posenew]
                            # the_file.write(str(posestrnew[0]) + ' '  + str(posestrnew[1]) + ' '  + str(posestrnew[2]) + ' '  + str(posestrnew[3]) + ' '  + str(posestrnew[4]) + ' '  + str(posestrnew[5]) + ' '  + str(posestrnew[6])+'\n')                
                           
                        # namenew=line.rstrip('\n').split('.')[0]
                        namenew=name.replace('.png','')+str("{:.3f}".format(x))+str("{:.3f}".format(y))+".png"

                        r_pose_t = np.asarray(posenew[0:3])-np.asarray(anchor_pose[0:3])
                        r_pose_quat = get_rotation_diff(anchor_pose[3:7], posenew[3:7]) 
                        r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                        r_pose = [float("{:.5f}".format(a)) for a in r_pose]
                        
                        with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_extrapolated'+'.txt', 'a') as the_file:
                            output='/'+anchor_name + ' ' + '/'+namenew +' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6])
                            # the_file.write(output+'\n')
                            
                            print(itr)
                            print(anchor1_index)
                            print(anchor2_index)
                            print(anchor3_index)
                            print(anchor4_index)

                            anchor1name=names_ss[anchor1_index].replace(".pose.txt",".color.png")
                            anchor2name=names_ss[anchor2_index].replace(".pose.txt",".color.png")
                            anchor3name=names_ss[anchor3_index].replace(".pose.txt",".color.png")
                            anchor4name=names_ss[anchor4_index].replace(".pose.txt",".color.png")
       
                            t1_pose=poses_ss[anchor1_index][0:3]
                            print(t1_pose)
                            t1_pose = [float("{:.5f}".format(a)) for a in t1_pose]    
                            print(t1_pose)

                            t2_pose=poses_ss[anchor2_index][0:3]                                
                            t2_pose = [float("{:.5f}".format(a)) for a in t2_pose]    
                            
                            t3_pose=poses_ss[anchor3_index][0:3]                                
                            t3_pose = [float("{:.5f}".format(a)) for a in t3_pose]    

                            t4_pose=poses_ss[anchor4_index][0:3]                                
                            t4_pose = [float("{:.5f}".format(a)) for a in t4_pose]    
                            
                            treg_pose=posenew[0:3]                                
                            treg_pose = [float("{:.5f}".format(a)) for a in treg_pose]
                            
                            with open('/home/mzaffar/Documents/datasets/University/'+scenes_dict[selected_scene]+'/NN_7scenes_refsampled_rposes_eplr'+'.txt', 'a') as the_file:
                                output='/' + anchor1name + ' ' + '/' + anchor2name + ' ' + '/' + anchor3name + ' ' + '/' + anchor4name + ' ' + '/' + namenew + ' ' \
                                + str(t1_pose[0]) + ' '  + str(t1_pose[1]) + ' '  + str(t1_pose[2]) + ' '  \
                                + str(t2_pose[0]) + ' '  + str(t2_pose[1]) + ' '  + str(t2_pose[2]) + ' ' \
                                + str(t3_pose[0]) + ' '  + str(t3_pose[1]) + ' '  + str(t3_pose[2]) + ' ' \
                                + str(t4_pose[0]) + ' '  + str(t4_pose[1]) + ' '  + str(t4_pose[2]) + ' ' \
                                + str(treg_pose[0]) + ' '  + str(treg_pose[1]) + ' '  + str(treg_pose[2]) + ' ' + '\n'
                                the_file.write(output)
                        

            itr=itr+1
    print("The sequence used for upsampling was: ", training_sequences[sequence_to_use] )

elif (dataset=='escalator'):
    samplingfactor=50
    directory='/home/mzaffar/Documents/datasets/station_escalator/escalator/'
    train_names=[]
    train_poses=[]
    test_names=[]
    test_poses=[]
    
    def get_rotation_diff(r_bestmatchpose, r_qpose):
        r1 = R.from_quat(r_bestmatchpose)
        r2 = R.from_quat(r_qpose)
        
        r1 = r1.as_matrix()
        r2 = r2.as_matrix()
        rot_diff = np.matmul(np.transpose(r1), r2)
        rot_diff = R.from_matrix(rot_diff).as_quat()
        rot_diff=[rot_diff[3],rot_diff[0],rot_diff[1],rot_diff[2]]
        
        return rot_diff

##############for creating NN7scenes_refsampled_rposes_extrapolated , NN7scenes_refsampled_rposes_eplr and gtposes_train_extrapolated ################    
    with open(directory + '/filenames_train_ds50_noqueryanchors.txt', 'r') as f:  # the train plus test containall the images of the dataset
        for line in f:
            train_names.append(line.rstrip())
    with open(directory + '/gtposes_train_ds50_noqueryanchors.txt', 'r') as f:  
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            train_poses.append(pose)
            
    with open(directory + '/filenames_test_ds50_includesqueryanchors.txt', 'r') as f:  
        for line in f:
            test_names.append(line.rstrip())
    with open(directory + '/gtposes_test_ds50_includesqueryanchors.txt', 'r') as f:  
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            test_poses.append(pose)
#################            
            
    for itr, train_name in enumerate(train_names):
        with open(directory+'/filenames_train_ds50.txt', 'a') as the_file:
            print('')
            # the_file.write(train_name+'\n') # creating the reference sequence that doesn't contain the images seen at training time by CoPR

    for itr, train_pose in enumerate(train_poses):
        with open(directory+'/gtposes_train_ds50.txt', 'a') as the_file:
            out = str(train_pose[0]) + ' '  + str(train_pose[1]) + ' '  + str(train_pose[2]) + ' '  + str(train_pose[3]) + ' '  + str(train_pose[4]) + ' '  + str(train_pose[5]) + ' '  + str(train_pose[6])
            # the_file.write(out+'\n')

    for itr, test_name in enumerate(test_names):
        if (itr%samplingfactor==0):
            with open(directory+'/filenames_train_ds50.txt', 'a') as the_file:
                print('')
                # the_file.write(test_name+'\n') # creating the reference sequence that doesn't contain the images seen at training time by CoPR

    for itr, test_pose in enumerate(test_poses):
        if (itr%samplingfactor==0):
            with open(directory+'/gtposes_train_ds50.txt', 'a') as the_file:
                out = str(test_pose[0]) + ' '  + str(test_pose[1]) + ' '  + str(test_pose[2]) + ' '  + str(test_pose[3]) + ' '  + str(test_pose[4]) + ' '  + str(test_pose[5]) + ' '  + str(test_pose[6])
                # the_file.write(out+'\n')
#################
                
#################
    for itr, test_name in enumerate(test_names):
        if (itr%samplingfactor!=0):
            with open(directory+'/filenames_test_ds50.txt', 'a') as the_file:
                print('')
                # the_file.write(test_name+'\n') # creating the reference sequence that doesn't contain the images seen at training time by CoPR

    for itr, test_pose in enumerate(test_poses):
        if (itr%samplingfactor!=0):
            with open(directory+'/gtposes_test_ds50.txt', 'a') as the_file:
                out = str(test_pose[0]) + ' '  + str(test_pose[1]) + ' '  + str(test_pose[2]) + ' '  + str(test_pose[3]) + ' '  + str(test_pose[4]) + ' '  + str(test_pose[5]) + ' '  + str(test_pose[6])
                # the_file.write(out+'\n')

#################                    
    for itr, train_name in enumerate(train_names):
        anchor_name=train_name
        reg_name=test_names[itr]+'_regressed'
        anchor_pose=train_poses[itr]        

        reg_pose=train_poses[itr].copy()   
        temp=reg_pose[0]-1.8
        reg_pose[0]=temp
        reg_pose = [float("{:.5f}".format(a)) for a in reg_pose]
            
        r_pose_t = np.asarray(reg_pose[0:3])-np.asarray(anchor_pose[0:3])
        r_pose_quat = get_rotation_diff(anchor_pose[3:7], reg_pose[3:7]) 
        r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
        r_pose = [float("{:.5f}".format(a)) for a in r_pose]
        
        with open(directory+'/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
                    output1='/'+ anchor_name + ' ' + '/' + anchor_name + ' ' + str(0) + ' '  + str(0) + ' '  + str(0) + ' '  + str(1) + ' '  + str(0) + ' '  + str(0) + ' '  + str(0)
                    # print(output)
                    the_file.write(output1+'\n')
        with open(directory+'/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:                    
                    output2='/'+anchor_name + ' ' + '/'+reg_name +' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) 
                    # print(output)
                    the_file.write(output2+'\n')
        with open(directory+'gtposes_train_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:                   
                    output3=str(anchor_pose[0]) + ' '  + str(anchor_pose[1]) + ' '  + str(anchor_pose[2]) + ' '  + str(anchor_pose[3]) + ' '  + str(anchor_pose[4]) + ' '  + str(anchor_pose[5]) + ' '  + str(anchor_pose[6])
                    # print(output)
                    the_file.write(output3+'\n')
        with open(directory+'gtposes_train_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
                    output4=str(reg_pose[0]) + ' '  + str(reg_pose[1]) + ' '  + str(reg_pose[2]) + ' '  + str(reg_pose[3]) + ' '  + str(reg_pose[4]) + ' '  + str(reg_pose[5]) + ' '  + str(reg_pose[6])
                    # print(output)
                    the_file.write(output4+'\n')
                    
        print(itr)
        if (itr-6 < 0):
            anchor1_index=itr #of ref traj
            anchor2_index=itr #of query traj
            anchor3_index=itr+6 #of ref traj
            anchor4_index=itr+samplingfactor #of query traj
            print('here1') 

        elif (itr+6 > len(train_names)-1):
            if (itr+3 > len(train_names)-1):
                anchor1_index=itr-6  #of ref traj
                anchor2_index=itr-samplingfactor #of query traj
                anchor3_index=itr-3  #of ref traj
                anchor4_index=itr #of query traj
                print('here2') 
            else:
                anchor1_index=itr-6 #of ref traj
                anchor2_index=itr-samplingfactor #of query traj
                anchor3_index=itr #of ref traj
                anchor4_index=itr #of query traj
                print('here3') 

        else:
            anchor1_index=itr-6
            anchor2_index=itr-samplingfactor
            anchor3_index=itr
            anchor4_index=itr
            print('here4') 
                            
        anchor1name=train_names[anchor1_index]
        anchor2name=test_names[anchor2_index-(anchor2_index%samplingfactor)] # on query sequence I should only have access to divisible by 3 indices which are available at training time
        anchor3name=train_names[anchor3_index]
        anchor4name=test_names[anchor4_index-(anchor4_index%samplingfactor)]  # on query sequence I should only have access to divisible by 3 indices which are available at training time
   
        t1_pose=train_poses[anchor1_index][0:3]
        print(t1_pose)
        t1_pose = [float("{:.5f}".format(a)) for a in t1_pose]    
        print(t1_pose)

        t2_pose=test_poses[anchor2_index-(anchor2_index%samplingfactor)][0:3]                                
        t2_pose = [float("{:.5f}".format(a)) for a in t2_pose]    
        
        t3_pose=train_poses[anchor3_index][0:3]                                
        t3_pose = [float("{:.5f}".format(a)) for a in t3_pose]    
        
        t4_pose=test_poses[anchor4_index-(anchor4_index%samplingfactor)][0:3]                                
        t4_pose = [float("{:.5f}".format(a)) for a in t4_pose]    
        
        treg_pose=reg_pose[0:3]                                
        treg_pose = [float("{:.5f}".format(a)) for a in treg_pose]

        with open(directory+'/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
            output='/'+ train_name + ' ' + '/' + train_name + ' ' + '/' + train_name + ' ' + '/' + train_name + ' ' + '/' + train_name + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) \
                + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0)
            # print(output)
            the_file.write(output+'\n') 
                        
        with open(directory+'/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
            output='/' + anchor1name + ' ' + '/' + anchor2name + ' ' + '/' + anchor3name + ' ' + '/' + anchor4name + ' ' + '/' + reg_name + ' ' \
            + str(t1_pose[0]) + ' '  + str(t1_pose[1]) + ' '  + str(t1_pose[2]) + ' '  \
            + str(t2_pose[0]) + ' '  + str(t2_pose[1]) + ' '  + str(t2_pose[2]) + ' ' \
            + str(t3_pose[0]) + ' '  + str(t3_pose[1]) + ' '  + str(t3_pose[2]) + ' ' \
            + str(t4_pose[0]) + ' '  + str(t4_pose[1]) + ' '  + str(t4_pose[2]) + ' ' \
            + str(treg_pose[0]) + ' '  + str(treg_pose[1]) + ' '  + str(treg_pose[2]) + ' ' + '\n'
            the_file.write(output)
            
######################################
#for adding anchor point of query traj to extrapolation dataloaders

    for itr, testname in enumerate(test_names):
        if (itr%samplingfactor==0):
            with open(directory+'/NN_7scenes_refsampled_rposes_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
                        output1='/'+ testname + ' ' + '/' + testname + ' ' + str(0) + ' '  + str(0) + ' '  + str(0) + ' '  + str(1) + ' '  + str(0) + ' '  + str(0) + ' '  + str(0)
                        # print(output)
                        the_file.write(output1+'\n')

    for itr, testpose in enumerate(test_poses):
        if (itr%samplingfactor==0):
            with open(directory+'gtposes_train_extrapolated'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:                   
                        output3=str(testpose[0]) + ' '  + str(testpose[1]) + ' '  + str(testpose[2]) + ' '  + str(testpose[3]) + ' '  + str(testpose[4]) + ' '  + str(testpose[5]) + ' '  + str(testpose[6])
                        # print(output)
                        the_file.write(output3+'\n')

    for itr, testname in enumerate(test_names):
        if (itr%samplingfactor==0):
            with open(directory+'/NN_7scenes_refsampled_rposes_eplr'+'_x'+str(0.0)+'_y'+str(0.0)+'_ss'+str(0.0)+'.txt', 'a') as the_file:
                output='/'+ testname + ' ' + '/' + testname + ' ' + '/' + testname + ' ' + '/' + testname + ' ' + '/' + testname + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) \
                    + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0) + ' ' + str(0) + ' ' + str(0) + ' '  + str(0)
                # print(output)
                the_file.write(output+'\n')

#######################################
            
##############for creating db_all_median_hard_train and val ################
    with open(directory + '/filenames_train_ds50_noqueryanchors.txt', 'r') as f:  
        for line in f:
            train_names.append(line.rstrip())
    with open(directory + '/gtposes_train_ds50_noqueryanchors.txt', 'r') as f:  
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            train_poses.append(pose)
    with open(directory + '/filenames_test_ds50_includesqueryanchors.txt', 'r') as f:  
        for line in f:
            test_names.append(line.rstrip())
    with open(directory + '/gtposes_test_ds50_includesqueryanchors.txt', 'r') as f:  
        for line in f:
            pose = np.empty([7])
            line=line.rstrip()
            linesplit = line.split()
            for j in range(7):    
                pose[j] = float(linesplit[j])
            test_poses.append(pose)

    for itr, train_name in enumerate(train_names):
        if (itr%samplingfactor==0):
            anchor_name=train_name
            anchor_pose=train_poses[itr] 
            for itr2 in range(150):    
                if (itr+samplingfactor*itr2 < len(train_names)):
                    reg_name=test_names[itr+samplingfactor*itr2]
                    reg_pose=test_poses[itr+samplingfactor*itr2]        
                                
                    r_pose_t = np.asarray(reg_pose[0:3])-np.asarray(anchor_pose[0:3])
                    r_pose_quat = get_rotation_diff(anchor_pose[3:7], reg_pose[3:7]) 
                    r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                    r_pose = [float("{:.5f}".format(a)) for a in r_pose]
            
                    with open(directory+'/db_all_med_hard_train_station_escalator.txt', 'a') as the_file:
                                output2='/'+anchor_name + ' ' + '/'+reg_name +' ' + str('0') + ' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) 
                                # print(output)
                                # the_file.write(output2+'\n')
                            
            for itr2 in range(150):
                if (itr+samplingfactor*itr2 < len(train_names)):
                    reg_name=train_names[itr+samplingfactor*itr2]
                    reg_pose=train_poses[itr+samplingfactor*itr2]        
                                
                    r_pose_t = np.asarray(reg_pose[0:3])-np.asarray(anchor_pose[0:3])
                    r_pose_quat = get_rotation_diff(anchor_pose[3:7], reg_pose[3:7]) 
                    r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                    r_pose = [float("{:.5f}".format(a)) for a in r_pose]
            
                    with open(directory+'/db_all_med_hard_train_station_escalator.txt', 'a') as the_file:
                                output2='/'+anchor_name + ' ' + '/'+reg_name +' ' + str('0') + ' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) 
                                # print(output)
                                # the_file.write(output2+'\n')
                        
        else:
            anchor_name=train_name
            anchor_pose=train_poses[itr] 
            for itr2 in range(150): 
                if (itr+samplingfactor*itr2 < len(train_names)):
                    reg_name=test_names[itr+samplingfactor*itr2]
                    reg_pose=test_poses[itr+samplingfactor*itr2]        
                                
                    r_pose_t = np.asarray(reg_pose[0:3])-np.asarray(anchor_pose[0:3])
                    r_pose_quat = get_rotation_diff(anchor_pose[3:7], reg_pose[3:7]) 
                    r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                    r_pose = [float("{:.5f}".format(a)) for a in r_pose]
            
                    with open(directory+'/db_all_med_hard_valid_station_escalator.txt', 'a') as the_file:
                                output2='/'+anchor_name + ' ' + '/'+reg_name +' ' + str('0') + ' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) 
                                # print(output)
                                # the_file.write(output2+'\n')
                            
            for itr2 in range(150):  
                if (itr+samplingfactor*itr2 < len(train_names)):
                    reg_name=train_names[itr+samplingfactor*itr2]
                    reg_pose=train_poses[itr+samplingfactor*itr2]        
                                
                    r_pose_t = np.asarray(reg_pose[0:3])-np.asarray(anchor_pose[0:3])
                    r_pose_quat = get_rotation_diff(anchor_pose[3:7], reg_pose[3:7]) 
                    r_pose=np.concatenate((r_pose_t, r_pose_quat), axis=-1)
                    r_pose = [float("{:.5f}".format(a)) for a in r_pose]
            
                    with open(directory+'/db_all_med_hard_valid_station_escalator.txt', 'a') as the_file:
                                output2='/'+anchor_name + ' ' + '/'+reg_name + ' ' + str('0') + ' ' + str(r_pose[0]) + ' '  + str(r_pose[1]) + ' '  + str(r_pose[2]) + ' '  + str(r_pose[3]) + ' '  + str(r_pose[4]) + ' '  + str(r_pose[5]) + ' '  + str(r_pose[6]) 
                                # print(output)
                                # the_file.write(output2+'\n')