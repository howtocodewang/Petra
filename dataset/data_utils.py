import os
import re
import sys
import time
import shutil
import psutil
import pypinyin
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import scipy.io as io
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


SUPPORTED_FILE = 'EEG', '21E', 'PNT', 'LOG', '11D', 'CMT', 'CN2', 'EGF', 'EVT', 'VF2', 'BFT', 'm2t', 'VOR', 'PTN', 'TRD'

def pil_loader(path):
    # open path as file to avoid ResourceWarning
    # (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def list2txt(list, txt_path):
    if os.path.exists(txt_path):
        os.remove(txt_path)

    f = open(txt_path, "w")
    for it in list:
        f.write(it + '\n')
    f.close()


def txt2list(txt_path):
    dst_list = []
    with open(txt_path, 'r') as f:
        for line in f:
            dst_list.append(line.strip('\n'))
    return dst_list


def getFilelist(folder_path, ext=SUPPORTED_FILE):
    filelist = []
    for root, dirs, files in os.walk((folder_path)):
        for filename in files:
            if filename.endswith(ext) and not filename.startswith('.'):
                # print("root", root)
                # print("dirs", dirs)
                # print("files", filename)
                filepath = os.path.join(root, filename)
                filelist.append(filepath)

    return filelist


def cal_file_size(path, type="MB"):
    size = os.stat(path).st_size
    if type == "KB":
        kb_size = size / 1024
        return kb_size
    elif type == "MB":
        mb_size = size / 1024 / 1024
        return mb_size
    elif type == "GB":
        gb_size = size / 1024 / 1024 / 1024
        return gb_size
    else:
        return size


def createdir(path, overwrite=True):
    if os.path.exists(path) and len(path) == 0:
        pass
    elif os.path.exists(path) and len(path) != 0 and overwrite:
        shutil.rmtree(path)
        os.mkdir(path)
    elif os.path.exists(path) and len(path) != 0 and not overwrite:
        pass
    else:
        os.mkdir(path)


def cloneFolderstructure(src_path, dst_path):
    createdir(dst_path)
    for root, dirs, files in os.walk((src_path)):
        for dir in dirs:
            if not dir.startswith('D'):
                os.path.join(root, dir).replace(src_path, dst_path)
                os.mkdir(os.path.join(root, dir).replace(src_path, dst_path))
            else:
                pass

    return print("Successfully clone {} dictionary structure".format(src_path))


def createSoftlink(eeg_list, src, dst, patient_info=None, ext=SUPPORTED_FILE, sorted_by_patient=False):
    instance_list = []
    instance_info = defaultdict(list)
    for eeg_int_path in tqdm(eeg_list):
        eeg_file_name = eeg_int_path.split('/')[-1]
        eeg_size = cal_file_size(eeg_int_path)
        yyyy_mm = eeg_int_path.split('/')[-4]
        eeg_dst_folder = eeg_int_path.replace(src, dst)[:-4]
        instance_name = os.path.splitext(eeg_file_name)[0]

        createdir(eeg_dst_folder)
        createdir(os.path.join(eeg_dst_folder, 'vis_results'))

        instance_list.append(instance_name)
        instance_info[instance_name] = [eeg_file_name, eeg_size, yyyy_mm, eeg_int_path, eeg_dst_folder]

        for e in ext:
            rel_path = os.path.join(eeg_int_path[:-4] + '.' + e)
            if os.path.exists(rel_path):
                link_path = os.path.join(eeg_dst_folder, instance_name + '.' + e)
                os.symlink(rel_path, link_path)
            else:
                pass

    print("Successfully create soft links of all files in {} and sort by instance in {}".format(src, dst))

    eeg_total_size = 0
    for k, v in instance_info.items():
        instance_size = v[1]
        eeg_total_size += instance_size

    print("The size of all EEG files is {} TB".format(eeg_total_size / 1024 / 1024))

    if sorted_by_patient and not patient_info.empty:
        parent_path = os.path.dirname(dst)
        patient_folder = os.path.join(parent_path, 'Patient')
        createdir(patient_folder)

        for index, row in patient_info.iterrows():
            patient_eeg_folder = os.path.join(patient_folder, row['Patient Name'])
            if os.path.exists(patient_eeg_folder):
                pass
            else:
                os.mkdir(patient_eeg_folder)

            patient_instance_list = row['Rec Instances']
            for ins in patient_instance_list:
                date = instance_info[ins][2]
                eeg_date_folder = os.path.join(patient_eeg_folder, date)
                if os.path.exists(eeg_date_folder):
                    pass
                else:
                    os.mkdir(eeg_date_folder)

                patient_instance_folder = os.path.join(eeg_date_folder, ins)
                if os.path.exists(patient_instance_folder):
                    pass
                else:
                    os.mkdir(patient_instance_folder)

                instance_info[ins].append(patient_instance_folder)
                createdir(os.path.join(patient_instance_folder, 'vis_results'))

                for e in ext:
                    rel_path = os.path.join(instance_info[ins][-3][:-4] + '.' + e)
                    if os.path.exists(rel_path):
                        link_path = os.path.join(patient_instance_folder, ins + '.' + e)
                        os.symlink(rel_path, link_path)
                    else:
                        pass

        print("Successfully create soft links of all files in {} and sort by patient in {}".format(src, patient_folder))

    return instance_list, instance_info


def get_instance_list_from_txt(eeg_path_txt_file):
    new_eeg_list = txt2list(eeg_path_txt_file)
    instance_list = []
    for e_f in new_eeg_list:
        instance_id = e_f.split('/')[-1].split('.')[0]
        instance_list.append(instance_id)
    return instance_list


def contains_digit(str):
    return bool(re.search(r'\d', str))


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def is_contains_chinese(strs):
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fa5':
            return True
    return False


def create_dataframe_dict(key_list):
    info_dataframe = {}
    for k in key_list:
        info_dataframe[k] = []
    return info_dataframe


def is_contains_duplicate(tgt_list):
    duplicate_dict = dict(Counter(tgt_list))
    duplicate_list = [key for key, value in duplicate_dict.items() if value > 1]
    duplicate_times = {key: value for key, value in duplicate_dict.items() if value > 1}
    if len(duplicate_list) == 0:
        print("There is no duplicated instance")
    return duplicate_list, duplicate_times


def remove_duplicate(new_patient_info_xlsx=None, new_patient_info_df=None, saved_path=None):
    if new_patient_info_xlsx is not None and new_patient_info_df is None:
        new_patient_info_df = pd.read_excel(new_patient_info_xlsx, engine='openpyxl')
    elif new_patient_info_xlsx is not None and new_patient_info_df is not None:
        print("Wrong input")
    patient_list = new_patient_info_df['Patient Name']

    dpl_list, _ = is_contains_duplicate(patient_list)

    del_index_list = []
    for pat in dpl_list:
        pat_idx_list = [i for i, x in enumerate(patient_list) if x == pat]
        pat_idx_list.sort()
        # print(pat_idx_list)
        pat_ins_list = []
        for idx in pat_idx_list:
            pat_ins_list.append(new_patient_info_df.iloc[idx]['Rec Instances'])
            if pat_idx_list.index(idx) != 0:
                del_index_list.append(idx)

        # if isinstance(pat_ins_list[0], list):
        #     pat_all_ins = sum(pat_ins_list, [])
        #     new_patient_info_df.loc[pat_idx_list[0], 'Rec Instances'] = pat_all_ins
        #     print(new_patient_info_df.iloc[pat_idx_list[0]]['Rec Instances'])
        #     print('a')
        # elif isinstance(pat_ins_list[0], str):
        #     pat_all_ins = " ".join(pat_ins_list)
        #     # new_pat_info_xlsx.iloc[pat_idx_list[0]]= pat_all_ins
        #     new_patient_info_df.loc[pat_idx_list[0], 'Rec Instances'] = pat_all_ins

        pat_all_ins = " ".join(pat_ins_list)
        # new_pat_info_xlsx.iloc[pat_idx_list[0]]= pat_all_ins
        new_patient_info_df.loc[pat_idx_list[0], 'Rec Instances'] = pat_all_ins

        # print('a')

    clean_df = new_patient_info_df.drop(index=del_index_list).reset_index(drop=True)
    print("Successfully merge the same patient instances information")

    if saved_path is not None:
        clean_df.to_excel(saved_path)
        print("No duplicated patient name excel file has been saved in {}".format(saved_path))
    else:
        return clean_df


