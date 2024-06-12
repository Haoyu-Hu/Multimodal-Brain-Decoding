import numpy as np
import os
import pickle
import nibabel as nib
import h5py

def time_interval_split(meta_path=None, save_file=False, setting_save_path=None, meta_csv_path=None):
    """
    Function: Select time points with image presentation in design matrix
    ----------------
    Input:
    meta_path: path to a set of design matrices
    save_file: True if the output is needed to be stored offline
    setting_save_path: the path to store the output
    meta_csv_path: the path for reading the meta csv doc containing image information
    ----------------
    Output:
    des_store: a file containing dict of time points with image at each run
    """
    assert os.path.isdir(meta_path), 'Invalid design matrix!'
    
    if meta_csv_path:
        import pandas as pd
        csv_file = pd.read_csv(meta_csv_path, index_col=0)
    
    des_store = {}
    for design_matrix_path in os.listdir(meta_path):
        file_path = os.path.join(meta_path, design_matrix_path)
        des_file = np.genfromtxt(file_path, delimiter='\t')
        img_idx = des_file[np.nonzero(des_file)].tolist()
        if meta_csv_path:
            share_1000 = csv_file.loc[np.array(img_idx, dtype=int)-1]['shared1000']
            cocoid = csv_file.loc[np.array(img_idx, dtype=int)-1]['cocoId']
            cocosplit = csv_file.loc[np.array(img_idx, dtype=int)-1]['cocoSplit']
        time_points = [i for i, val in enumerate(des_file) if val]
        if not img_idx:
            des_store[design_matrix_path[-19:-4]] = ['rs_fmri']
        elif meta_csv_path:
            des_store[design_matrix_path[-19:-4]] = [time_points, np.array(img_idx, dtype=int)-1, np.array(cocoid.tolist(), dtype=int), cocosplit.tolist(), np.array(share_1000.tolist())]
        else:
            des_store[design_matrix_path[-19:-4]] = [time_points, np.array(img_idx, dtype=int)-1]
    
    if save_file:
        meta_mt_path, design_meta_path = os.path.split(meta_path)
        if not setting_save_path:
            new_path = os.path.join(meta_mt_path, 'selected_design_matrix')
        else:
            new_path = setting_save_path
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_file = os.path.join(new_path, 'selected_design_matrix.pkl')
        with open(new_file, 'wb') as f:
            pickle.dump(des_store, f)
    
    return des_store

def fmri_split(meta_path=None, design_matrix=None, split=1, save_file=False, setting_save_path=None, setting_test_path=None):
    """
    Function: Split fmri image (3D) data with processed design matrix
    -------------------
    Input:
    meta_path: path to a set of nii.gz file from NSD dataset
    design_matrix: either path to processed design matrix or file
    save_file: True to save the output
    setting_save_path: the path to store the output
    -------------------
    Output:
    fmri_store: a list containing all fmri with image presentation
    """
    assert os.path.isdir(meta_path), 'Invalid fmri data path!'
    
    if isinstance(design_matrix, dict):
        des_mat = design_matrix
    elif os.path.isfile(design_matrix):
        with open(design_matrix, 'rb') as f:
            des_mat = pickle.load(f)
    else:
        raise Exception('Unknown design matrix')
    
    fmri_store = []
    test_store = []
    session_count = int(os.listdir(meta_path)[-1][-15:-13])
    split_patch = session_count // split
    split_count = 1
    
    for sub_file in sorted(os.listdir(meta_path)):
        if str(split_patch*split_count+1) in sub_file[-22:-13]:
            print('------Split {} Finished-------'.format(split_count))
            if save_file:
                print('Saving File for Split {}...'.format(split_count))
                meta_mt_path, fmri_meta = os.path.split(meta_path)
                if not setting_save_path:
                    new_path = os.path.join(meta_mt_path, 'selected_fmri')
                else:
                    new_path = setting_save_path
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                new_file = os.path.join(new_path, 'selected_fmri_'+str(split_count)+'.pkl')
                with open(new_file, 'wb') as f:
                    pickle.dump(np.concatenate(fmri_store, axis=0), f)
            split_count += 1
            del fmri_store
            fmri_store = []
    
        print('---FMRI_SPLIT: PROCESSING {}------'.format(sub_file))
        fmri_file_path = os.path.join(meta_path, sub_file)
        fmri_file = nib.load(fmri_file_path)
        fmri_array = np.transpose(fmri_file.get_fdata(), (-1, 0, 1, 2))
        del fmri_file
        time_points = des_mat[sub_file[-22:-7]][0]
        share1000 = des_mat[sub_file[-22:-7]][-1]
        if time_points == 'rs_fmri':
            continue
        fmri_select = fmri_array[time_points]
        del fmri_array
        fmri_train_val = fmri_select[~share1000]
        fmri_test = fmri_select[share1000]
        del fmri_select
        fmri_store.append(fmri_train_val)
        del fmri_train_val
        test_store.append(fmri_test)
        del fmri_test
        
    if save_file:
        print('Saving File for Split {}...'.format(split_count))
        meta_mt_path, fmri_meta = os.path.split(meta_path)
        if not setting_save_path:
            new_path = os.path.join(meta_mt_path, 'selected_fmri')
        else:
            new_path = setting_save_path
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_file = os.path.join(new_path, 'selected_fmri_'+str(split_count)+'.pkl')
        with open(new_file, 'wb') as f:
            pickle.dump(np.concatenate(fmri_store, axis=0), f)
        del fmri_store
        if not setting_test_path:
            test_path = new_path
        else:
            test_path = setting_test_path
        test_file = os.path.join(test_path, 'test.pkl')
        print('Saving Test File')
        with open(test_file, 'wb') as f:
            pickle.dump(np.concatenate(test_store, axis=0), f)
    
    print('Done!')
    
    return test_store  
    

def find_img(img_path=None, design_matrix=None, capt_set=None, split=1, save_file=False, setting_save_path=None, setting_test_path=None):
    """
    Function: locate corresponding images according to processed design matrix
    ----------------------
    Input:
    img_path: path to the hdf5 file storing images (nsd_stimuli.hdf5)
    design_matrix: processed design matrix by function 'time_interval_split' or the path to it
    ------------------------
    Output:
    img_list: a list of image selected, in which the first element is the image set, the second is whether it belongs to shared1000 set
    """
    
    assert os.path.isfile(img_path), 'Invalid NSD Experiment Image File Path'
    
    if isinstance(design_matrix, dict):
        des_mat = design_matrix
    elif os.path.isfile(design_matrix):
        with open(design_matrix, 'rb') as f:
            des_mat = pickle.load(f)
    else:
        raise Exception('Unknown design matrix')
    
    if capt_set:
        from pycocotools.coco import COCO
        coco_train = COCO(capt_set[0])
        coco_val = COCO(capt_set[1])
    
    img_store = []
    test_store = []
    capt = []
    capt_test = []
    session_count = int(list(des_mat.keys())[-1][-8:-6])
    split_patch = session_count // split
    split_count = 1
    
    for sub_file in sorted(list(des_mat.keys())):
        if str(split_patch*split_count+1) in sub_file[:-6]:
            print('------Split {} Finished-------'.format(split_count))
            if save_file:
                print('Saving File for Split {}...'.format(split_count))
                img_meta, img_file = os.path.split(img_path)
                if not setting_save_path:
                    new_path = os.path.join(img_meta, 'selected_img')
                else:
                    new_path = setting_save_path
                if not os.path.isdir(new_path):
                    os.mkdir(new_path)
                new_file = os.path.join(new_path, 'selected_img_'+str(split_count)+'.pkl')
                with open(new_file, 'wb') as f:
                    pickle.dump(np.concatenate(img_store, axis=0), f)
                if capt_set:
                    cap_file = os.path.join(capt_set[2], 'selected_capt_'+str(split_count)+'.pkl')
                    with open(cap_file, 'wb') as f:
                        pickle.dump(capt, f)
            split_count += 1
            del img_store
            img_store = []
    
        print('---FIND_IMAGE: PROCESSING {}------'.format(sub_file))
        share1000 = des_mat[sub_file][-1]
        cocoid = des_mat[sub_file][2]
        cocosplit = des_mat[sub_file][3]
        if des_mat[sub_file][0] == 'rs_fmri':
            continue
        img_idx = des_mat[sub_file][1]
        with h5py.File(img_path, 'r') as img_set:
            img_select = []
            capt_select = []
            for idx_img in img_idx:
                img_select.append(img_set['imgBrick'][idx_img])
                if capt_set:
                    coco_id = cocoid[idx_img]
                    coco_split = cocosplit[idx_img]
                    if 'train' in coco_split:
                        capIds = coco_train.getAnnIds(imgIds=coco_id)
                        caps = coco_train.loadAnns(capIds)
                        capt_select.append(caps)
                    else:
                        capIds = coco_val.getAnnIds(imgIds=coco_id)
                        caps = coco_val.loadAnns(capIds)
                        capt_select.append(caps)
            img_select = np.stack(img_select, axis=0)
            img_train_val = img_select[~share1000]
            img_test = img_select[share1000]
            del img_select
            img_store.append(img_train_val)
            del img_train_val
            test_store.append(img_test)
            del img_test
            if capt_set:
                cap_train_val = capt_select[~share1000]
                cap_test = capt_select[share1000]
                capt.append(cap_train_val)
                capt_test.append(cap_test)
        
    if save_file:
        print('Saving File for Split {}...'.format(split_count))
        img_meta, img_file = os.path.split(img_path)
        if not setting_save_path:
            new_path = os.path.join(img_meta, 'selected_img')
        else:
            new_path = setting_save_path
        if not os.path.isdir(new_path):
            os.mkdir(new_path)
        new_file = os.path.join(new_path, 'selected_img_'+str(split_count)+'.pkl')
        with open(new_file, 'wb') as f:
            pickle.dump(img_store, f)
        if capt_set:
            cap_file = os.path.join(capt_set[2], 'selected_capt_'+str(split_count)+'.pkl')
            with open(cap_file, 'wb') as f:
                pickle.dump(capt, f)
            cap_test_file = os.path.join(capt_set[3], 'test.pkl')
            with open(cap_test_file, 'wb') as f:
                pickle.dump(capt_test, f)
        if not setting_test_path:
            test_path = new_path
        else:
            test_path = setting_test_path
        test_file = os.path.join(test_path, 'test.pkl')
        print('Saving Test File')
        with open(test_file, 'wb') as f:
            pickle.dump(np.concatenate(test_store, axis=0), f)
    
    print('Done!')
    
    return img_store, test_store

def data_integration(fmri_path=None, img_path=None, capt_path=None, save_path=None):
    """
    """
    fmri_file = os.listdir(fmri_path)
    img_file = os.listdir(img_path)
    capt_file = os.listdir(capt_path)

    for idx_file in range(len(fmri_file)):
        print('Integrating {}, {}, {}'.format(fmri_file[idx_file], img_file[idx_file], capt_file[idx_file]))
        store_list = []

        with open(os.path.join(fmri_path, fmri_file[idx_file]), 'rb') as f:
            fmri = pickle.load(f)
        with open(os.path.join(img_path, img_file[idx_file]), 'rb') as f:
            img = pickle.load(f)
        with open(os.path.join(capt_path, capt_file[idx_file]), 'rb') as f:
            capt = pickle.load(f)
        
        store_dict = {}
        for idx_in_file in range(fmri.shape[0]):
            store_dict['fmri'] = fmri[idx_in_file]
            store_dict['image'] = img[idx_in_file]
            store_dict['caption'] = capt[idx_in_file]
            store_list.append(store_dict)
    
        print('Saving File for Split {}'.format(str(idx_file+1)))
        save_file = os.path.join(save_path, 'integrated_data_'+str(idx_file+1)+'.pkl')
        with open(save_file, 'wb') as f:
            pickle.dump(store_list, f)
    print('Done!')

    return 1