import numpy as np
import constants as cst
import os
from torch.utils import data
import torch


def _build_segment_indices(segments, seq_size, slice_start, slice_len, return_relative=True, stock_offset=0):
    """Return valid start indices and stock ids for a slice.

    segments: list of dicts with start/end (absolute indices in full file)
    seq_size: required window length
    slice_start: absolute start idx of this slice in the full file
    slice_len: length of this slice
    return_relative: if True, subtract slice_start so indices align to the slice array
    stock_offset: base id to add for this block (useful when concatenating files)
    """
    slice_end = slice_start + slice_len - 1
    starts = []
    stock_ids = []
    for i, seg in enumerate(segments):
        seg_start = int(seg["start"])
        seg_end = int(seg["end"])
        inter_start = max(seg_start, slice_start)
        inter_end = min(seg_end, slice_end)
        if inter_start > inter_end:
            continue
        last_start = inter_end - seq_size + 1
        if last_start < inter_start:
            continue
        rng = np.arange(inter_start, last_start + 1, dtype=np.int64)
        if return_relative:
            rng = rng - slice_start
        starts.append(rng)
        stock_ids.append(np.full_like(rng, stock_offset + i, dtype=np.int64))
    if not starts:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    return np.concatenate(starts), np.concatenate(stock_ids)

  
def fi_2010_load(path, seq_size, horizon, all_features, enforce_segments=False):
    dec_data = np.loadtxt(path + "/Train_Dst_NoAuction_ZScore_CF_7.txt")
    
    train_segments_list = []
    val_segments_list = []
    
    train_meta = cst.FI2010_TRAIN["Train_Dst_NoAuction_ZScore_CF_7.txt"]
    segments = train_meta["segments"]
    split_rate = cst.SPLIT_RATES[0]
    
    for seg in segments:
        start = int(seg["start"])
        end = int(seg["end"]) + 1 
        seg_data = dec_data[:, start:end]
        seg_len = seg_data.shape[1]
        
        split_idx = int(seg_len * split_rate)
        
        train_segments_list.append(seg_data[:, :split_idx])
        val_segments_list.append(seg_data[:, split_idx:])
        
    full_train = np.hstack(train_segments_list)
    full_val = np.hstack(val_segments_list)

    dec_test1 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_7.txt')
    dec_test2 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_8.txt')
    dec_test3 = np.loadtxt(path + '/Test_Dst_NoAuction_ZScore_CF_9.txt')
    full_test = np.hstack((dec_test1, dec_test2, dec_test3))

    if horizon == 10:
        tmp = 5
    elif horizon == 20:
        tmp = 4
    elif horizon == 30:
        tmp = 3
    elif horizon == 50:
        tmp = 2
    elif horizon == 100:
        tmp = 1
    else:
        raise ValueError("Horizon not found")
    
    train_labels = full_train[-tmp, :].flatten()
    val_labels = full_val[-tmp, :].flatten()
    test_labels = full_test[-tmp, :].flatten()
    
    train_labels = train_labels[seq_size-1:] - 1
    val_labels = val_labels[seq_size-1:] - 1
    test_labels = test_labels[seq_size-1:] - 1
    if all_features:
        train_input = full_train[:144, :].T
        val_input = full_val[:144, :].T
        test_input = full_test[:144, :].T
    else:
        train_input = full_train[:40, :].T
        val_input = full_val[:40, :].T
        test_input = full_test[:40, :].T

    # Optional: build per-stock indices so sequences don't cross segments
    train_indices = val_indices = test_indices = None
    train_stock_ids = val_stock_ids = test_stock_ids = None
    if enforce_segments:
        train_meta = cst.FI2010_TRAIN["Train_Dst_NoAuction_ZScore_CF_7.txt"]
        raw_segments = train_meta["segments"]
        split_rate = cst.SPLIT_RATES[0]
        
        train_idx_list = []
        train_stock_list = []
        current_offset = 0
        
        for i, seg in enumerate(raw_segments):
            seg_start = int(seg["start"])
            seg_end = int(seg["end"]) + 1
            full_seg_len = seg_end - seg_start
            
            seg_train_len = int(full_seg_len * split_rate)
            
            block_idx, block_stock = _build_segment_indices(
                [{"start": 0, "end": seg_train_len - 1}], 
                seq_size, 
                slice_start=0, 
                slice_len=seg_train_len, 
                return_relative=True, 
                stock_offset=i 
            )
            
            block_idx += current_offset
            
            train_idx_list.append(block_idx)
            train_stock_list.append(block_stock)
            
            current_offset += seg_train_len

        train_indices = np.concatenate(train_idx_list)
        train_stock_ids = np.concatenate(train_stock_list)

        val_idx_list = []
        val_stock_list = []
        current_offset = 0
        
        for i, seg in enumerate(raw_segments):
            seg_start = int(seg["start"])
            seg_end = int(seg["end"]) + 1
            full_seg_len = seg_end - seg_start
            
            seg_train_len = int(full_seg_len * split_rate)
            seg_val_len = full_seg_len - seg_train_len
            
            block_idx, block_stock = _build_segment_indices(
                [{"start": 0, "end": seg_val_len - 1}], 
                seq_size, 
                slice_start=0, 
                slice_len=seg_val_len, 
                return_relative=True, 
                stock_offset=i
            )
            
            block_idx += current_offset
            
            val_idx_list.append(block_idx)
            val_stock_list.append(block_stock)
            
            current_offset += seg_val_len

        val_indices = np.concatenate(val_idx_list)
        val_stock_ids = np.concatenate(val_stock_list)

        test_meta1 = cst.FI2010_TEST["Test_Dst_NoAuction_ZScore_CF_7.txt"]
        test_meta2 = cst.FI2010_TEST["Test_Dst_NoAuction_ZScore_CF_8.txt"]
        test_meta3 = cst.FI2010_TEST["Test_Dst_NoAuction_ZScore_CF_9.txt"]
        
        test1_len = dec_test1.shape[1]
        test2_len = dec_test2.shape[1]
        test3_len = dec_test3.shape[1]
        
        test_idx1, stock1 = _build_segment_indices(
            test_meta1["segments"], seq_size, slice_start=0, slice_len=test1_len, return_relative=True, stock_offset=0
        )
        
        test_idx2, stock2 = _build_segment_indices(
            test_meta2["segments"], seq_size, slice_start=0, slice_len=test2_len, return_relative=True, stock_offset=0
        )
        test_idx2 += test1_len
        
        test_idx3, stock3 = _build_segment_indices(
            test_meta3["segments"], seq_size, slice_start=0, slice_len=test3_len, return_relative=True, stock_offset=0
        )
        test_idx3 += (test1_len + test2_len)

        test_indices = np.concatenate((test_idx1, test_idx2, test_idx3))
        test_stock_ids = np.concatenate((stock1, stock2, stock3))

    train_input = torch.from_numpy(train_input).float()
    train_labels = torch.from_numpy(train_labels).long()
    val_input = torch.from_numpy(val_input).float()
    val_labels = torch.from_numpy(val_labels).long()
    test_input = torch.from_numpy(test_input).float()
    test_labels = torch.from_numpy(test_labels).long()

    if not enforce_segments:
        return train_input, train_labels, val_input, val_labels, test_input, test_labels

    train_indices = torch.from_numpy(train_indices).long()
    val_indices = torch.from_numpy(val_indices).long()
    test_indices = torch.from_numpy(test_indices).long()
    train_stock_ids = torch.from_numpy(train_stock_ids).long()
    val_stock_ids = torch.from_numpy(val_stock_ids).long()
    test_stock_ids = torch.from_numpy(test_stock_ids).long()

    return (
        train_input,
        train_labels,
        val_input,
        val_labels,
        test_input,
        test_labels,
        train_indices,
        val_indices,
        test_indices,
        train_stock_ids,
        val_stock_ids,
        test_stock_ids,
    )
