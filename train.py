import torch
import torch.nn as nn
import torch.nn.functional as F

import argparse
from tqdm import tqdm

from models import FreePointTransformer, HeatRegNet
from point_utils import transform_points, generate_mesh, resample_sweep, random_rotation, get_kpts_support

import os


'''
Training Routine
Please adapt for your own pruposes.

We rely on 3 types of data:
* kptss_radius_fixed_tf -> Straight Sweeps (fixed)
* kptss_radius_moving_tf -> non-tracked C/S-Sweeps (moving)
* meshs_radius_target_tf -> tracked C/S-Sweeps (target /Ground Truth)


'''


def random_coords(batched_kpts, n=1024):
    b,_ , _ = batched_kpts.shape
    max_coords = batched_kpts.view(b,-1,3).max(1)[0].unsqueeze(1)
    min_coords = batched_kpts.view(b,-1,3).min(1)[0].unsqueeze(1)
    random_coords = torch.rand(b, n, 3).to(batched_kpts.device) * (max_coords - min_coords) + min_coords
    return random_coords


def grid_coords(batched_kpts):
    b, n, _ = batched_kpts.shape
    max_coords = batched_kpts.view(b,-1,3).max(1,keepdim=True)[0]
    min_coords = batched_kpts.view(b,-1,3).min(1,keepdim=True)[0]
    grids = []
    for i in range(b):
        grid = torch.meshgrid([torch.linspace(min_coords[i, 0, 0], max_coords[i, 0, 0], 8),
                               torch.linspace(min_coords[i, 0, 1], max_coords[i, 0, 1], 4),
                               torch.linspace(min_coords[i, 0, 2], max_coords[i, 0, 2], 32)])
        grid = torch.stack(grid, dim=-1).view(-1, 3)
        grids.append(grid)
    grid_batchwise = torch.stack(grids, dim=0).to(batched_kpts.device)
    return grid_batchwise



def train_one_epoch(model, optimizer,  bs, subjects_train, meshs_radius_fixed_tf, kptss_support_fixed_tf_resampled, all_moving, all_target, num_points_resample, num_frames, random_aug = True, no_support=False, support = None):
    model.train()

    batches = torch.tensor([(subject, sweep) for subject in subjects_train for sweep in [0,1]])
    batches = batches[torch.randperm(batches.size(0))][:(batches.size(0)//bs)*bs].view(-1, bs, 2)
    loss_batch = 0

    for batch in batches:
        optimizer.zero_grad()
        subjects = batch[:, 0]
        
        sweeps_moving = sweeps_target = batch[:, 1]
        if support=='random':
            kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[batch,0].view(bs,-1,3),random_coords(kptss_radius_fixed_tf[subjects,0].view(bs,-1,3), 1024)], dim=1)
        elif support=='grid':
            kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[batch,0].view(bs,-1,3),grid_coords(kptss_radius_fixed_tf[subjects,0].view(bs,-1,3))], dim=1)

        
        
        
        
        kpts_radius_moving_tf = all_moving[subjects, sweeps_moving]
        kpts_radius_target_tf = all_target[subjects, sweeps_target]

        if random_aug:
            random_rot = random_rotation(bs).to(device)
            kpts_radius_fixed_tf = transform_points(random_rot, kpts_radius_fixed_tf.view(bs, -1, 3)).view(bs, num_frames, -1, 3)
            kpts_radius_moving_tf = transform_points(random_rot, kpts_radius_moving_tf.view(bs, -1, 3)).view(bs, num_frames, -1, 3)
            kpts_radius_target_tf = transform_points(random_rot, kpts_radius_target_tf.view(bs, -1, 3)).view(bs, num_frames, -1, 3)

        disp = model(kpts_radius_moving_tf.view(bs, -1, 3), kpts_radius_fixed_tf.view(bs, -1, 3)).view(bs, num_frames, -1, 3)
        kpts_radius_pred_tf_ = kpts_radius_moving_tf + disp

        T_radius_disp = torch.eye(4).unsqueeze(0).repeat(bs*num_frames, 1, 1).to(device)
        T_radius_disp[:, :3, 3] = (kpts_radius_pred_tf_.view(bs*num_frames, -1, 3)-kpts_radius_moving_tf.view(bs*num_frames, -1, 3)).mean(1)
        kpts_radius_pred_tf = transform_points(T_radius_disp, kpts_radius_moving_tf.view(bs*num_frames, -1, 3)).view(bs, num_frames, -1, 3)

        loss = (F.smooth_l1_loss(kpts_radius_target_tf, kpts_radius_pred_tf) + F.smooth_l1_loss(kpts_radius_target_tf, kpts_radius_pred_tf_))/2

        loss.backward()
        optimizer.step()
        loss_batch += loss.item()
    return loss_batch/len(batches)


def validate_after_one_epoch(model, subject_val, meshs_radius_fixed_tf, kptss_support_fixed_tf_resampled, all_moving, all_target, num_points_resample, num_frames, no_support=False, bs=2, support = None):
    model.eval()
    batch = torch.tensor([(subject_val, sweep) for sweep in [0,1]]).view(-1, 2)
    with torch.no_grad():
        subjects = batch[:, 0]
        sweeps_moving = sweeps_target = batch[:, 1]

        if support=='random':
            kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[batch,0].view(bs,-1,3),random_coords(kptss_radius_fixed_tf[subjects,0].view(bs,-1,3), 1024)], dim=1)
        elif support=='grid':
            kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[batch,0].view(bs,-1,3),grid_coords(kptss_radius_fixed_tf[subjects,0].view(bs,-1,3))], dim=1)

        kpts_radius_moving_tf = all_moving[subjects, sweeps_moving]
        kpts_radius_target_tf = all_target[subjects, sweeps_target]

        disp = model(kpts_radius_moving_tf.view(2, -1, 3), kpts_radius_fixed_tf.view(2, -1, 3)).view(2, num_frames, -1, 3)
        kpts_radius_pred_tf_ = kpts_radius_moving_tf + disp

        T_radius_disp = torch.eye(4).unsqueeze(0).repeat(2*num_frames, 1, 1).to(device)
        T_radius_disp[:, :3, 3] = (kpts_radius_pred_tf_.view(2*num_frames, -1, 3)-kpts_radius_moving_tf.view(2*num_frames, -1, 3)).mean(1)
        kpts_radius_pred_tf = transform_points(T_radius_disp, kpts_radius_moving_tf.view(2*num_frames, -1, 3)).view(2, num_frames, -1, 3)

        loss = (F.smooth_l1_loss(kpts_radius_target_tf, kpts_radius_pred_tf) + F.smooth_l1_loss(kpts_radius_target_tf, kpts_radius_pred_tf_))/2
    return loss.item()


def create_new_moving_pts(model, num_frames, num_points_resample, moving_pts, no_support=False, support=None):
    model.eval()
    with torch.no_grad():
        subjects = torch.arange(21).view(-1, 1).repeat(1, 2).view(-1)
        
        kpts_radius_pred_tf = torch.zeros(42, 64, 32, 3)
        for i in range(21):

            if support=='random':
                kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[i,0].view(1,-1,3),random_coords(kptss_radius_fixed_tf[i,0].view(1,-1,3), 1024)], dim=1).repeat(2, 1, 1)
            elif support=='grid':
                kpts_radius_fixed_tf = torch.cat([kptss_radius_fixed_tf[i,0].view(1,-1,3),grid_coords(kptss_radius_fixed_tf[i,0].view(1,-1,3))], dim=1).repeat(2,1,1)
        
            kpts_radius_moving_tf = moving_pts[i:i+1, torch.tensor([0, 1])]

            disp = model(kpts_radius_moving_tf.view(2, -1, 3), kpts_radius_fixed_tf.view(2, -1, 3)).view(2, num_frames, -1, 3)
            kpts_radius_pred_tf_ = kpts_radius_moving_tf + disp

            T_radius_disp = torch.eye(4).unsqueeze(0).repeat(2*num_frames, 1, 1).to(device)
            T_radius_disp[:, :3, 3] = (kpts_radius_pred_tf_.view(2*num_frames, -1, 3)-kpts_radius_moving_tf.view(2*num_frames, -1, 3)).mean(1)
            kpts_radius_pred_tf[i*2:(i+1)*2] = transform_points(T_radius_disp, kpts_radius_moving_tf.view(2*num_frames, -1, 3)).view(2, num_frames, -1, 3)
    
    return kpts_radius_pred_tf.view(21,2,64,32,3)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=3)
    parser.add_argument('--subject_test', '-st', type=int, required=True)
    parser.add_argument('-k', type=int, default=64)
    parser.add_argument('--base', type=int, default=24)
    parser.add_argument('--batch_sz', type=int, default=4)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--num_frames', type=int, default=64)
    parser.add_argument('--num_points_resample', type=int, default=2048)
    parser.add_argument('--epochs_FPT', type=int, default=5000)
    parser.add_argument('--epochs_HRN', type=int, default=3000)
    parser.add_argument('--save_path', type=str, default='saved_models')
    parser.add_argument('--pre_computed_fpt', type=str, default='None')
    parser.add_argument('--no_support', action='store_true', default=False)
    parser.add_argument('--support', type=str, default=None, choices=['random', 'grid'])
    args = parser.parse_args()

    print(args)
    
    ##set device
    device='cuda'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

    ##prepare data
    subjects_test = torch.tensor([args.subject_test])
    subjects_val = torch.tensor([(args.subject_test-1)%21])
    print('Subjects Test:', subjects_test, 'Subjects Val:', subjects_val)
    
    subjects_train = torch.tensor([i for i in range(21) if i not in subjects_test and i not in subjects_val])
    print('Subjects Train:', subjects_train)

    dict_to_save = {'args': args, 
                    'subjects_test': subjects_test,
                    'subjects_val': subjects_val,
                    'subjects_train': subjects_train}
    


    data = torch.load('data.pth')
    
    kptss_radius = data['kptss_radius']
    Ts_gt_radius = data['Ts_gt_radius']
    Ts_init_radius = data['Ts_init_radius']

    for subject in range(21):
        for sweep in range(4):
            keep = []
            for i in range(kptss_radius[subject][sweep].shape[0]):
                if kptss_radius[subject][sweep][i].unique(dim=0).shape[0] >= 16:
                    keep.append(i)
            ind = torch.linspace(0,kptss_radius[subject][sweep][keep].shape[0]-1,args.num_frames).round().int()
            kptss_radius[subject][sweep] = kptss_radius[subject][sweep][keep][ind]
            Ts_gt_radius[subject][sweep] = Ts_gt_radius[subject][sweep][keep][ind]
            Ts_init_radius[subject][sweep] = Ts_init_radius[subject][sweep][keep][ind]

    kptss_radius_fixed_tf = torch.stack([torch.stack([transform_points(Ts_gt_radius[subject][sweep], kptss_radius[subject][sweep]) for sweep in [0]]) for subject in range(21)])
    kptss_radius_moving_tf = torch.stack([torch.stack([transform_points(Ts_init_radius[subject][sweep], kptss_radius[subject][sweep]) for sweep in [1, 2]]) for subject in range(21)])
    kptss_radius_target_tf = torch.stack([torch.stack([transform_points(Ts_gt_radius[subject][sweep], kptss_radius[subject][sweep]) for sweep in [1, 2]]) for subject in range(21)])


    if args.no_support or args.support!=None:
        meshs_radius_fixed_tf = None
        meshs_radius_target_tf = None
        kptss_support_fixed_tf_resampled = None
    else:
        meshs_radius_fixed_tf = [[generate_mesh(kptss_radius_fixed_tf[subject][sweep]) for sweep in [0]] for subject in tqdm(range(21))]
        meshs_radius_target_tf = [[generate_mesh(kptss_radius_target_tf[subject][sweep]) for sweep in [0, 1]] for subject in tqdm(range(21))]

        kptss_support_fixed_tf_resampled = torch.stack([
                                                torch.stack([
                                                    get_kpts_support(kptss_radius_fixed_tf[subject, 0], torch.stack([torch.stack([resample_sweep(meshs_radius_target_tf[subject][sweep], 1024) for sweep in [0, 1]]) for subject in subjects_train]), 1024)
                                                for sweep in [0]])
                                            for subject in tqdm(range(21))])

    if args.pre_computed_fpt == 'None':
        ##stage 1: Free Point Transformer
        fpt_model = FreePointTransformer().to(device)
        optimizer = torch.optim.Adam(fpt_model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2*args.epochs_FPT//3])

        loss_fpt_train = []
        loss_fpt_val = []

        for i in tqdm(range(args.epochs_FPT)):
            loss = train_one_epoch(
                            model=fpt_model,
                            optimizer=optimizer,
                            bs=args.batch_sz,
                            subjects_train= subjects_train,
                            meshs_radius_fixed_tf=meshs_radius_fixed_tf,
                            kptss_support_fixed_tf_resampled=kptss_support_fixed_tf_resampled,
                            all_moving = kptss_radius_moving_tf,
                            all_target= kptss_radius_target_tf,
                            num_points_resample=args.num_points_resample,
                            num_frames=args.num_frames,
                            random_aug = True,
                            no_support=args.no_support,
                            support=args.support)
            
            loss_fpt_train.append(loss)
            scheduler.step()
            loss = validate_after_one_epoch(
                                    model=fpt_model,
                                    subject_val=subjects_val,
                                    meshs_radius_fixed_tf=meshs_radius_fixed_tf,
                                    kptss_support_fixed_tf_resampled=kptss_support_fixed_tf_resampled,
                                    all_moving=kptss_radius_moving_tf,
                                    all_target=kptss_radius_target_tf,
                                    num_points_resample=args.num_points_resample,
                                    num_frames=args.num_frames,
                                    no_support=args.no_support,
                                    support=args.support)
            loss_fpt_val.append(loss)
            if i%50 == 0 and i>0:
                tqdm.write(f'FPT Iter {i}, loss_train: {torch.tensor(loss_fpt_train[-1]).mean().item()}, loss_test: {loss_fpt_val[-1]}')

        #create and save new moving pts
        torch.save(fpt_model.state_dict(), args.save_path+f'/fpt_model_{args.subject_test}.pth')
        kpts_radius_pred_tf_fpt = create_new_moving_pts(fpt_model, args.num_frames, args.num_points_resample, kptss_radius_moving_tf, args.no_support)
        torch.save(kpts_radius_pred_tf_fpt.cpu(), args.save_path+f'/kpts_radius_pred_tf_fpt_{args.subject_test}.pth')

    
    else:
        print('Using pre-computed FPT Points')
        kpts_radius_pred_tf_fpt = torch.load(args.pre_computed_fpt).to(device)
        print('Loaded pre-computed FPT Points')


    ##stage 2: HeatRegNet
    kpts_radius_pred_tf_fpt = kpts_radius_pred_tf_fpt.to(device)
    hrn_model = HeatRegNet(args.k, args.stride, args.base).to(device)
    optimizer = torch.optim.SGD(hrn_model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [args.epochs_HRN//2,args.epochs_HRN//4*3])



    loss_hrn_train = []
    loss_hrn_val = []
    

    loss_min = 10000
    for i in tqdm(range(args.epochs_HRN)):
        loss = train_one_epoch( model=hrn_model,
                                optimizer=optimizer,
                                bs=args.batch_sz,
                                subjects_train= subjects_train,
                                meshs_radius_fixed_tf=meshs_radius_fixed_tf,
                                kptss_support_fixed_tf_resampled=kptss_support_fixed_tf_resampled,
                                all_moving = kpts_radius_pred_tf_fpt,
                                all_target= kptss_radius_target_tf,
                                num_points_resample=args.num_points_resample,
                                num_frames=args.num_frames,
                                random_aug = False,
                                no_support=args.no_support,
                                support=args.support
        )
        loss_hrn_train.append(loss)
        scheduler.step()


        val_loss = validate_after_one_epoch(
                                model=hrn_model,
                                subject_val=subjects_val,
                                meshs_radius_fixed_tf=meshs_radius_fixed_tf,
                                kptss_support_fixed_tf_resampled=kptss_support_fixed_tf_resampled,
                                all_moving=kpts_radius_pred_tf_fpt,
                                all_target=kptss_radius_target_tf,
                                num_points_resample=args.num_points_resample,
                                num_frames=args.num_frames,
                                no_support=args.no_support,
                                support=args.support)
                                
        loss_hrn_val.append(val_loss)

        if i%50 == 0 and i>0:
            tqdm.write(f'HRN Iter {i}, loss_train: {torch.tensor(loss_hrn_train[-1]).mean().item()}, loss_test: {loss_hrn_val[-1]}')
        if val_loss < loss_min:
            tqdm.write(f'New best model found at iter {i}, loss: {val_loss}')
            loss_min = loss_hrn_val[-1]
            torch.save(hrn_model.state_dict(), args.save_path+f'/hrn_model_{args.subject_test}_best.pth')

            dict_to_save['best_iter'] = i
            dict_to_save['loss_hrn_train'] = loss_hrn_train
            dict_to_save['loss_hrn_val'] = loss_hrn_val

            torch.save(dict_to_save, args.save_path+f'/dict_{args.subject_test}.pth')

            #save args dict + losses + iter of best model together
            #add best_iter to args dict to save

    #save final model
    torch.save(hrn_model.state_dict(), args.save_path+f'/hrn_model_{args.subject_test}_final.pth')
    # create and save new moving pts
    #load best model
    hrn_model.load_state_dict(torch.load(args.save_path+f'/hrn_model_{args.subject_test}_best.pth'))
    kpts_radius_pred_tf_hrn = create_new_moving_pts(hrn_model, args.num_frames, args.num_points_resample, kpts_radius_pred_tf_fpt, args.no_support, support=args.support)
    torch.save(kpts_radius_pred_tf_hrn.cpu(), args.save_path+f'/kpts_radius_pred_tf_hrn_{args.subject_test}.pth')









