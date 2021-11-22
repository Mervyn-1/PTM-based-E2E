import sys
import torch
import random
import numpy as np
import json
from torch.nn.utils import rnn
import progressbar
import random
import json
from torch import nn
import os

def map_bool(bool_status):
    if bool_status == 'True': 
        return True
    elif bool_status == 'False': 
        return False
    else:
        raise Exception('Wrong Bool Status')

format_mapping_dict = {
    'metalwoz': {'nlu': False, 'bs': False, 'da': False, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'kvret': {'nlu': False, 'bs': True, 'da': False, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'woz': {'nlu': False, 'bs': True, 'da': False, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'camres676': {'nlu': False, 'bs': True, 'da': False, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'taskmaster': {'nlu': False, 'bs': True, 'da': False, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'e2e_ms': {'nlu': False, 'bs': True, 'da': True, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'frames': {'nlu': False, 'bs': True, 'da': True, 'nlg': True, 'turn_info': True, 'spans_prediction': True},
    'schema_guided': {'nlu': False, 'bs': True, 'da': True, 'nlg': True, 'turn_info': True, 'spans_prediction': True}
}

dataset_name_list = ['e2e_ms', 'metalwoz', 'kvret', 'woz', 'camres676', 'taskmaster', 'frames', 'schema_guided']

class TOD_PRETRAINING_CORPUS:
    def __init__(self, tokenizer, shuffle_mode, dataset_prefix_path, use_nlu, use_bs, use_da, use_nlg, use_turn_info, use_spans_prediction, max_tgt_len=128):
        self.use_nlu, self.use_bs, self.use_da, self.use_nlg, self.use_turn_info, self.use_spans_prediction = \
        map_bool(use_nlu), map_bool(use_bs), map_bool(use_da), map_bool(use_nlg), map_bool(use_turn_info), map_bool(use_spans_prediction)
        print ('use NLU: {}, use DST: {}, use POL: {}, use NLG: {}, use Turn Info: {}, use Spans Prediction: {}'.format(use_nlu, use_bs, use_da, use_nlg, use_turn_info, use_spans_prediction))

        print ('Tokenizer Size is %d' % len(tokenizer))
        self.tokenizer = tokenizer
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(['<_PAD_>'])[0]
        self.sos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<sos_context>'])[0]
        self.eos_context_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_context>'])[0]

        self.eos_b_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_b>'])[0]
        self.eos_a_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_a>'])[0]
        self.eos_r_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_r>'])[0]
        self.eos_d_token_id = self.tokenizer.convert_tokens_to_ids(['<eos_d>'])[0]

        self.shuffle_mode = shuffle_mode
        self.max_tgt_len = max_tgt_len

        # construct task-specific prefix
        bs_prefix_text = 'translate dialogue to belief state:'
        self.bs_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(bs_prefix_text))
        da_prefix_text = 'translate dialogue to dialogue action:'
        self.da_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(da_prefix_text))
        nlg_prefix_text = 'translate dialogue to system response:'
        self.nlg_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(nlg_prefix_text))
        ic_prefix_text = 'translate dialogue to user intent:'
        self.ic_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ic_prefix_text))
        ti_prefix_text = 'reconstruct dialogue turn information:'
        self.ti_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(ti_prefix_text))
        sp_prefix_text = 'reconstruct corrupted input text:'
        self.sp_prefix_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(sp_prefix_text))

        self.train_all_dataset_list = self.load_data(dataset_prefix_path, train_test_mode='train', \
            use_bs=self.use_bs, use_da=self.use_da, use_nlg=self.use_nlg, use_turn_info=self.use_turn_info, use_spans_prediction=self.use_spans_prediction)
        self.dev_all_dataset_list = self.load_data(dataset_prefix_path, train_test_mode='test', \
            use_bs=self.use_bs, use_da=self.use_da, use_nlg=self.use_nlg, use_turn_info=self.use_turn_info, use_spans_prediction=self.use_spans_prediction)

        if self.use_nlu == True:
            print ('Add Intent Classification for Pretraining.')
            train_ic_sess_list, dev_ic_sess_list = self.load_train_dev_intent_classification_data(dataset_prefix_path)
        elif self.use_nlu == False:
            print ('Do not Add Intent Classification for Pretraining.')
            train_ic_sess_list, dev_ic_sess_list = [], []
        else:
            raise Exception('Wrong use_Intent_Classification Mode!!!')
        self.train_all_dataset_list += train_ic_sess_list
        self.dev_all_dataset_list += dev_ic_sess_list

        train_session_num, dev_session_num = len(self.train_all_dataset_list), len(self.dev_all_dataset_list)

        self.train_data_list = self.shuffle_train_data()
        self.dev_data_list = self.flatten_data(self.dev_all_dataset_list, mode='dev_set')
        print ('train session number is {}, train turn number is {}, train turn number per session {}'.format(train_session_num, len(self.train_data_list), round(len(self.train_data_list)/train_session_num, 2)))
        print ('dev session number is {}, dev turn numbder is {}, dev turn number per session {}'.format(dev_session_num, len(self.dev_data_list), round(len(self.dev_data_list)/dev_session_num, 2)))
        self.train_num, self.dev_num = len(self.train_data_list), len(self.dev_data_list)

    def load_data(self, dataset_prefix_path, train_test_mode, use_bs, use_da, use_nlg, use_turn_info, use_spans_prediction):
        all_dataset_list = []
        for name in dataset_name_list:
            one_dataset_list = self.parse_one_dataset(dataset_prefix_path, name, train_test_mode, use_bs, use_da, use_nlg, use_turn_info, use_spans_prediction)
            if len(one_dataset_list) > 0:
                all_dataset_list += one_dataset_list
        return all_dataset_list

    def parse_one_dataset(self, dataset_prefix_path, data_set_name, train_test_mode, use_bs, use_da, use_nlg, use_turn_info, use_spans_prediction):
        assert train_test_mode in ['train', 'test']
        # train_test_mode: 'train' or 'test'
        bs_exist, da_exist, nlg_exist, turn_info_exist, spans_predictions_exist = format_mapping_dict[data_set_name]['bs'], \
        format_mapping_dict[data_set_name]['da'], format_mapping_dict[data_set_name]['nlg'], format_mapping_dict[data_set_name]['turn_info'], format_mapping_dict[data_set_name]['spans_prediction']
        dataset_path = dataset_prefix_path + '/' + data_set_name + '_' + train_test_mode + '.json'

        print ('Loading data from {}'.format(dataset_path))
        with open(dataset_path) as f:
            data = json.load(f) 

        all_sess_list = []
        for one_sess in data:
            dial_sess_list = one_sess["dialogue_session"] # this list contains all turns from on session
            one_sess_list = [] 
            # one_sess_list is a list of turns
            # each turn is list of tuple pairs
            previous_context = []
            previous_context_without_special_tokens = []
            turn_num = len(dial_sess_list)
            for turn_id in range(turn_num):
                curr_turn = dial_sess_list[turn_id]
                curr_turn_list = []
                # this is a list of tuple pair (src, tgt)
                # [(nlg_input, nlg_output), (bs_input, bs_output), (da_input, da_output)]
                curr_user_input = curr_turn['user_id_list']
                curr_sys_resp = curr_turn['resp_id_list']
                # ----------------------------------------------------------- #
                if use_nlg and nlg_exist: # adding nlg data into pre-training procedure
                    # construct nlg_input, nlg_output
                    nlg_input = previous_context + curr_user_input 
                    nlg_input = self.nlg_prefix_id + [self.sos_context_token_id] + \
                    nlg_input[-900:] + [self.eos_context_token_id]
                    nlg_output = curr_sys_resp[:-1][:self.max_tgt_len] + [self.eos_r_token_id] # constrain the maximum tgt len
                    curr_turn_list.append((nlg_input, nlg_output))

                if use_bs and bs_exist:
                    bs_input = previous_context + curr_user_input
                    bs_input = self.bs_prefix_id + [self.sos_context_token_id] + bs_input[-900:] + \
                    [self.eos_context_token_id]
                    curr_bspn = curr_turn['bspn_id_list']
                    bs_output = curr_bspn[:-1][:self.max_tgt_len] + [self.eos_b_token_id]
                    curr_turn_list.append((bs_input, bs_output))

                if use_da and da_exist:
                    da_input = previous_context + curr_user_input 
                    da_input = self.da_prefix_id + [self.sos_context_token_id] + da_input[-900:] + \
                        [self.eos_context_token_id]
                    curr_aspn = curr_turn['aspn_id_list']
                    da_output = curr_aspn[:-1][:self.max_tgt_len] + [self.eos_a_token_id]
                    curr_turn_list.append((da_input, da_output))

                if use_turn_info and turn_info_exist:
                    # 重建user、response的special tokens
                    user_turn_info_input = previous_context_without_special_tokens + curr_user_input[1:-1] + curr_sys_resp[1:-1]
                    user_turn_info_input = self.ti_prefix_id + [self.sos_context_token_id] + user_turn_info_input[-900:] + \
                        [self.eos_context_token_id]
                    user_turn_info_output = previous_context + curr_user_input + curr_sys_resp
                    user_turn_info_output = user_turn_info_output[-900:]
                    curr_turn_list.append((user_turn_info_input, user_turn_info_output))

                if use_spans_prediction and spans_predictions_exist:
                    # spans predictions
                    noise_density = 0.15
                    mean_noise_span_length = 3
                    current_dialog_history = previous_context + curr_user_input + curr_sys_resp
                    current_dialog_history = current_dialog_history[-900:]
                    length = len(current_dialog_history)

                    num_noise_tokens = int(round(len(current_dialog_history) * noise_density))
                    num_noise_tokens = min(max(num_noise_tokens, 1), length)
                    num_noise_spans = int(round(num_noise_tokens / mean_noise_span_length))
                    num_noise_spans = max(num_noise_spans, 1)
                    num_nonnoise_tokens = length - num_noise_tokens

                    noise_span_lengths = self._random_segmentation(num_noise_tokens, num_noise_spans)
                    nonnoise_span_lengths = self._random_segmentation(num_nonnoise_tokens, num_noise_spans)

                    interleaved_span_lengths = np.reshape(
                        np.stack([nonnoise_span_lengths, noise_span_lengths], axis=1), [num_noise_spans * 2]
                    )

                    span_starts = np.cumsum(interleaved_span_lengths)[:-1]
                    span_start_indicator = np.zeros((length,), dtype=np.int8)
                    span_start_indicator[span_starts] = True
                    span_num = np.cumsum(span_start_indicator)
                    mask_indices = np.equal(span_num % 2, 1)
                    labels_mask = ~mask_indices

                    input_ids_sentinel = self.create_sentinel_ids(mask_indices.astype(np.int8))
                    labels_mask = self.create_sentinel_ids(labels_mask.astype(np.int8))

                    span_prediction_input = list(self.filter_input_ids(current_dialog_history, input_ids_sentinel))
                    span_prediction_output = list(self.filter_input_ids(current_dialog_history, labels_mask))

                    span_prediction_input = self.sp_prefix_id + [self.sos_context_token_id] + span_prediction_input + [self.eos_context_token_id]

                    curr_turn_list.append((span_prediction_input, span_prediction_output))

                if len(curr_turn_list) > 0:
                    one_sess_list.append(curr_turn_list)
                # update previous context
                previous_context = previous_context + curr_user_input + curr_sys_resp
                previous_context_without_special_tokens = previous_context_without_special_tokens + \
                    curr_user_input[1:-1] + curr_sys_resp[1:-1]

            if len(one_sess_list) > 0:
                all_sess_list.append(one_sess_list)
        return all_sess_list

    def filter_input_ids(self, input_ids, sentinel_ids):
        """
        Puts sentinel mask on `input_ids` and fuse consecutive mask tokens into a single mask token by deleting.
        This will reduce the sequence length from `expanded_inputs_length` to `input_length`.
        """
        input_ids_full = np.where(sentinel_ids != 0, sentinel_ids, input_ids)
        input_ids = input_ids_full[input_ids_full > 0]
        return input_ids

    def _random_segmentation(self, num_items, num_segments):
        """Partition a sequence of items randomly into non-empty segments.
        Args:
            num_items: an integer scalar > 0
            num_segments: an integer scalar in [1, num_items]
        Returns:
            a Tensor with shape [num_segments] containing positive integers that add
            up to num_items
        """
        mask_indices = np.arange(num_items - 1) < (num_segments - 1)
        np.random.shuffle(mask_indices)
        first_in_segment = np.pad(mask_indices, [[1, 0]])
        segment_id = np.cumsum(first_in_segment)
        # count length of sub segments assuming that list is sorted
        _, segment_length = np.unique(segment_id, return_counts=True)
        return segment_length

    def create_sentinel_ids(self, mask_indices):
        """
        Sentinel ids creation given the indices that should be masked.
        The start indices of each mask are replaced by the sentinel ids in increasing
        order. Consecutive mask indices to be deleted are replaced with `-1`.
        """
        start_indices = mask_indices - np.roll(mask_indices, 1, axis=-1) * mask_indices
        start_indices[0] = mask_indices[0]

        sentinel_ids = np.where(start_indices != 0, np.cumsum(start_indices, axis=-1), start_indices)
        sentinel_ids = np.where(sentinel_ids != 0, (self.tokenizer.convert_tokens_to_ids('<extra_id_0>') + 1 - sentinel_ids), 0)
        sentinel_ids -= mask_indices - start_indices

        return sentinel_ids

    def load_intent_classification_data(self, path):
        
        '''
            we treat each instance as a session, each data instance is treated as one session with one turn.
            so all instances have the following format
            [
                [
                    [(ic_src, ic_tgt)]
                ],

                [
                    [(ic_src, ic_tgt)]
                ],

                ...
            ]
        '''
        print ('Loading data from {}'.format(path))
        all_sess_list = []
        with open(path) as f:
            data = json.load(f) 
        for one_dict in data:
            one_intent_input = self.ic_prefix_id + [self.sos_context_token_id] + \
                one_dict['user_id_list'][-900:] + [self.eos_context_token_id]
            one_intent_output = one_dict['intent_id_list']
            one_turn = [(one_intent_input, one_intent_output)]
            one_sess = [one_turn]
            all_sess_list.append(one_sess)
        return all_sess_list

    def load_train_dev_intent_classification_data(self, dataset_prefix_path):
        train_path = dataset_prefix_path + '/train_intent_classification.json'
        train_ic_sess_list = self.load_intent_classification_data(train_path)
        dev_path = dataset_prefix_path + '/test_intent_classification.json'
        dev_ic_sess_list = self.load_intent_classification_data(dev_path)
        return train_ic_sess_list, dev_ic_sess_list

    def shuffle_train_data(self):
        return self.flatten_data(self.train_all_dataset_list, mode='train_set')

    def flatten_data(self, all_dataset_session_list, mode):
        '''
            all_dataset_session_list: 
                contains all sessions from all datasets
                each session contains multiple turns
                each turn is a list which has a format ranging from 
                    [(nlg_input, nlg_output)], 
                    [(nlg_input, nlg_output), (bs_input, bs_output)],
                    [(nlg_input, nlg_output), (bs_input, bs_output), (da_input, da_output)]
                    [(nlg_input, nlg_output), (bs_input, bs_output), (da_input, da_output), (turn_info_input, turn_info_output)]
        '''
        flatten_data_list = []
        if mode == 'train_set':
            if self.shuffle_mode == 'session_level':
                tmp_session_list = all_dataset_session_list.copy()
                random.shuffle(tmp_session_list)
                for one_session in tmp_session_list:
                    for one_turn in one_session:
                        for one_tuple in one_turn:
                            flatten_data_list.append(one_tuple)
            elif self.shuffle_mode == 'turn_level':
                for one_session in all_dataset_session_list:
                    for one_turn in one_session:
                        for one_tuple in one_turn:
                            flatten_data_list.append(one_tuple)
                random.shuffle(flatten_data_list)
            else:
                raise Exception('Wrong Shuffle Mode!!!')
        elif mode == 'dev_set':
            for one_session in all_dataset_session_list:
                for one_turn in one_session:
                    for one_tuple in one_turn:
                        flatten_data_list.append(one_tuple)
        else:
            raise Exception()
        return flatten_data_list

    def get_batches(self, batch_size, mode):
        #batch_size = self.cfg.batch_size
        batch_list = []
        if mode == 'train':
            self.train_data_list = self.shuffle_train_data()
            all_data_list = self.train_data_list
        elif mode == 'dev':
            all_data_list = self.dev_data_list
        else:
            raise Exception('Wrong Mode!!!')

        all_input_data_list, all_output_data_list = [], []
        for inp, oup in all_data_list:
            all_input_data_list.append(inp)
            all_output_data_list.append(oup)

        data_num = len(all_input_data_list)
        batch_num = int(data_num/batch_size) + 1

        for i in range(batch_num):
            start_idx, end_idx = i*batch_size, (i+1)*batch_size
            if start_idx > data_num - 1:
                break
            end_idx = min(end_idx, data_num - 1)
            one_input_batch_list, one_output_batch_list = [], []
            for idx in range(start_idx, end_idx):
                one_input_batch_list.append(all_input_data_list[idx])
                one_output_batch_list.append(all_output_data_list[idx])
            one_batch = [one_input_batch_list, one_output_batch_list]
            batch_list.append(one_batch)
        out_str = 'Overall Number of datapoints is ' + str(data_num) + \
        ' Number of ' + mode + ' batches is ' + str(len(batch_list))
        print (out_str)
        return batch_list

    def build_iterator(self, batch_size, mode):
        batch_list = self.get_batches(batch_size, mode)
        for i, batch in enumerate(batch_list):
            yield batch

    def pad_batch(self, batch_id_list):
        batch_id_list = [torch.LongTensor(item) for item in batch_id_list]
        batch_tensor = rnn.pad_sequence(batch_id_list, batch_first=True, padding_value=self.pad_token_id)
        batch_mask = torch.ones_like(batch_tensor)
        batch_mask = batch_mask.masked_fill(batch_tensor.eq(self.pad_token_id), 0.0).type(torch.FloatTensor)
        return batch_tensor, batch_mask

    def process_output(self, batch_tgt_id_list):
        batch_tgt_id_list = [torch.LongTensor(item) for item in batch_tgt_id_list]
        batch_tgt_tensor, _ = self.pad_batch(batch_tgt_id_list)
        batch_tgt_input_tensor = batch_tgt_tensor[:, :-1].clone()
        batch_tgt_output_tensor = batch_tgt_tensor[:, 1:].clone()
        return batch_tgt_input_tensor, batch_tgt_output_tensor

    def parse_batch_tensor(self, batch):
        batch_input_id_list, batch_output_id_list = batch
        batch_src_tensor, batch_src_mask = self.pad_batch(batch_input_id_list)
        batch_input, batch_labels = self.process_output(batch_output_id_list)
        batch_labels[batch_labels[:, :] == self.pad_token_id] = -100
        return batch_src_tensor, batch_src_mask, batch_input, batch_labels

