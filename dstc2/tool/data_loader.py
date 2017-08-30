# coding:utf-8

'''
  This script is for data loading.
  Created on Aug 8, 2017
  Author: qihu@mobvoi.com
'''

import copy
import numpy as np
import tensorflow as tf


# Read the knowledge base
def read_kb_value(kb_path):
    with open(kb_path) as f:
        lines = f.readlines()
    names = []
    entities = []
    values = {'R_cuisine': [],
              'R_location': [],
              'R_price': [],
              'R_post_code': [],
              'R_address': [],
              'R_phone': []}
    val2attr = {}
    num_line = len(lines)
    i = 0
    while i < num_line:
        line = lines[i].strip('\n').split(' ')
        name = line[1]
        if name not in names:
            names.append(name)
            entity = {'name': name}
            name0 = name
        while name == name0:
            attribute = line[2]
            value = line[3]
            entity[attribute] = value
            if value not in values[attribute]:
                values[attribute].append(value)
                val2attr[value] = attribute
            i += 1
            if i >= num_line:
                break
            line = lines[i].strip('\n').split(' ')
            name = line[1]
        entities.append(entity)
    return names, values, val2attr, entities


# Read dialog data and return user utterance & system response in two list
# Argument:
#   data_path: path of data file
# Return:
#   usr_list: a list of user dialog, for each dialog there are several utterances
#   sys_list: a list of user dialog, for each dialog there are several utterances
#   api_call_list: a list of api_call results, each one is composed of：
#       index: position of api_call in original data file
#       dialog_id
#       turn_id
#       names: a list of restaurants
#       num_restaurant: number of restaurants
def read_dialog(data_path):
    usr_list = []
    sys_list = []
    api_call_list = []
    with open(data_path) as f:
        lines = f.readlines()
    num_line = len(lines)
    dialog_id = 0
    turn_id = 0
    usr_dialog_list = []
    sys_dialog_list = []
    i = 0
    while i < num_line:
        line = lines[i].strip('\n')
        if len(line) == 0:
            usr_list.append(usr_dialog_list)
            sys_list.append(sys_dialog_list)
            usr_dialog_list = []
            sys_dialog_list = []
            dialog_id += 1
            turn_id = 0
            i += 1
            continue
        line = line.split(' ', 1)[1].split('\t')
        if len(line) == 1:  # Results of api_call
            api_call = {'index': i-1,
                        'dialog_id': dialog_id,
                        'turn_id': turn_id-1,
                        'names': [],
                        'attributes': [],
                        'num_restaurant': 0
                        }
            if line[0] != 'api_call no result':  # One or more restaurants found
                while len(line) == 1:  # Restaurant found
                    line = line[0].split(' ')
                    name = line[0]
                    attribute = line[1]
                    value = line[2]
                    if name not in api_call['names']:
                        api_call['names'].append(name)
                        api_call['num_restaurant'] += 1
                        api_call['attributes'].append({'name': name})
                    api_call['attributes'][api_call['num_restaurant']-1][attribute] = value
                    i += 1
                    line = lines[i].strip('\n').split(' ', 1)[1].split('\t')
                i -= 1
            api_call_list.append(api_call)
        else:  # A pair of ordinary utterances
            usr = line[0]
            sys = line[1]
            usr_dialog_list.append(usr)
            sys_dialog_list.append(sys)
            turn_id += 1
        i += 1
    # print('\tNumber of dialog: %d' % dialog_id)
    return usr_list, sys_list, api_call_list


# Generate a list of api_call result number
# (the length of this list is equal to total number of turns)
def get_kb_result(api_call_list, num_turn):
    kb_result_list = []
    for a in api_call_list:
        print 0


# Extract utterances from dialog sub-list
def extract_utc(utc_list):
    utc_plain_list = []
    for dialog in utc_list:
        for utc in dialog:
            utc_plain_list.append(utc)
    return utc_plain_list


# Read word2id_dict from vocab
def read_word2id(vocab_path, num_word):
    word2id_dict = {}
    id2word_list = ['']
    with open(vocab_path) as f:
        lines = f.readlines()
    for i in range(num_word):
        word = lines[i].split(' ')[0]
        word2id_dict[word] = i+1
        id2word_list.append(word)
    return word2id_dict, id2word_list


# Convert words to ids (all utc are in the same "big" list)
def convert_str2id(utc_list, word2id_dict, names, val2attr, max_length):
    id_list = []
    max_actual_length = 0
    num_utc = len(utc_list)

    for i in range(num_utc):
        utc = utc_list[i]
        id_dialog_list = []
        line = utc.strip('\n').split(' ')
        line = ['<s>'] + line + ['</s>']
        id_vector = np.zeros(max_length)
        max_actual_length = max(len(line), max_actual_length)
        actual_length = min(len(line), max_length)
        for j in range(actual_length):
            word = line[j]
            if word in names:  # replace restaurant name
                word = 'R_name'
            if word in val2attr:  # replace restaurant attributes
                word = val2attr[word]
            id_vector[j] = word2id_dict[word]
        id_dialog_list.append(id_vector)
    id_list.append(id_dialog_list)
    # print('\tMax length: %d' % max_actual_length)
    return id_list


# Convert words to ids (input utc_list is a 2D list, each element represents a dialogue)
def convert_2D_str2id(utc_list, word2id_dict, names, val2attr, max_length, back=False, add_headrear=False):
    id_list = []
    max_actual_length = 0
    num_dialog = len(utc_list)

    for i in range(num_dialog):
        dialog = utc_list[i]
        id_dialog_list = []
        num_turn = len(dialog)
        for j in range(num_turn):
            line = dialog[j].strip('\n').split(' ')
            if add_headrear:
                line = ['<s>'] + line + ['</s>']
            id_vector = np.zeros(max_length)
            max_actual_length = max(len(line), max_actual_length)
            actual_length = min(len(line), max_length)
            start = max_length - actual_length
            for j in range(actual_length):
                word = line[j]
                if word not in word2id_dict.keys():
                    continue
                if word in names:  # replace restaurant name
                    word = 'R_name'
                if word in val2attr:  # replace restaurant attributes
                    word = val2attr[word]
                if back:
                    id_vector[start+j] = word2id_dict[word]
                else:
                    id_vector[j] = word2id_dict[word]
            id_dialog_list.append(id_vector)
        id_list.append(id_dialog_list)
    # print('\tMax length: %d' % max_actual_length)
    return id_list


# Generate input data with all history (in word_id format)
def merge_dialog(usr_list, sys_list, max_turn):
    dialog_list = []
    num_dialog = len(usr_list)
    for i in range(num_dialog):
        # print i
        dialog = []
        history = ['']*2*max_turn
        usr_dialog = usr_list[i]
        sys_dialog = sys_list[i]
        num_turn = len(usr_dialog)
        for j in range(num_turn):
            usr_utc = usr_dialog[j]
            sys_utc = sys_dialog[j]
            history = history[-2*max_turn:]
            history.append(usr_utc)
            # print len(history)
            dialog.append(copy.copy(history))
            history.append(sys_utc)
        dialog_list.append(dialog)
    return dialog_list


# Get the number of restaurant in api_call result
def get_api_number(api_call_list, dialog_list):
    api_num = len(api_call_list)
    num_dialog = len(dialog_list)
    api_number_list = []
    for i in range(num_dialog):
        dialog = dialog_list[i]
        num_turn = len(dialog)
        api_number = []
        num_restaurant = -1
        for j in range(num_turn):
            api_number.append(num_restaurant)
            for k in range(api_num):
                if api_call_list[k]['dialog_id'] == i and api_call_list[k]['turn_id'] == j:
                    num_restaurant = api_call_list[k]['num_restaurant']
                    break
        api_number_list.append(api_number)
    return api_number_list


# Convert each history(a list of recent utc) into a single string
def flatten_history(dialog_list):
    num_dialog = len(dialog_list)
    for i in range(num_dialog):
        dialog = dialog_list[i]
        num_turn = len(dialog)
        for j in range(num_turn):
            whole_trun = ''
            turn = dialog[j]
            num_utc = len(turn)
            for k in range(num_utc):
                if turn[k] != '':
                    utc = '<s> %s </s>' % turn[k]
                    whole_trun = whole_trun + utc + ' '
            dialog[j] = whole_trun[:-1]
    return dialog_list


# flatten a 2D list into 1D (save all element to a "big" list)
def flatten_2D(data_list):
    num_data = len(data_list)
    flat_list = []
    for i in range(num_data):
        data = data_list[i]
        num_element = len(data)
        for j in range(num_element):
            flat_list.append(data[j])
    return flat_list


# restore model for test
def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables() if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x:x.name.split(':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

# Read sys template for LG classification
def read_sys_template(data_path):
    template_dict = {}
    with open(data_path) as f:
        lines = f.readlines()
    num_line = len(lines)
    for i in range(num_line):
        line = lines[i].strip('\n').split('\t')
        index = line[0]
        sentence = line[1]
        template_dict[sentence] = index
    return template_dict


#
def convert_template(s, name_list, val2attr):
    words = s.split(' ')
    num_word = len(words)
    values = val2attr.keys()
    for i in range(num_word):
        if words[i] in name_list:
            words[i] = 'R_name'
        elif words[i] in values:
            words[i] = val2attr[words[i]]
    return ' '.join(words)


def get_template_label(data_path, template_path, kb_path):
    names, values, val2attr, entities = read_kb_value(kb_path)
    usr_list, sys_list, api_call_list = read_dialog(data_path)
    template_dict = read_sys_template(template_path)
    template_keys = template_dict.keys()
    template_label_list = []
    sys_list = flatten_2D(sys_list)
    for i in range(len(sys_list)):
        s = sys_list[i]
        s = convert_template(s, names, val2attr)
        if s in template_keys:
            template_label_list.append(int(template_dict[s]))
        else:
            template_label_list.append(s)
            print i, s
    return template_label_list


if __name__ == '__main__':
    data_path = 'data/dialog-babi-task6-data-all.txt'
    vocab_path = 'data/all_vocab.txt'
    template_path = 'data/template/sys_resp_template.labeled.txt'
    kb_path = 'data/dialog-babi-task6-data-kb.txt'
    names, values, val2attr, entities = read_kb_value(kb_path)
    usr_list, sys_list, api_call_list = read_dialog(data_path)
    # usr_plain_list = extract_utc(usr_list)
    # sys_plain_list = extract_utc(sys_list)
    # word2id_dict, id2word_list = read_word2id(vocab_path, 770)
    # id_list = convert_str2id(usr_plain_list, word2id_dict, names, val2attr, 20)
    # dialog = merge_dialog(usr_list, sys_list, 2)
    # dialog = flatten_history(dialog)
    # dialog_id = convert_2D_str2id(dialog, word2id_dict, names, val2attr, 5*20)
    # api_number_list = get_api_number(api_call_list, dialog)
    template_dict = read_sys_template(template_path)
    label_list = get_template_label(data_path, template_path, kb_path)

    # # Test read_dialog
    # for i in range(10):  # Show data
    #     num_turn = len(usr_list[i])
    #     for j in range(num_turn):
    #         print usr_list[i][j], '*'*5,  sys_list[i][j]
    #     print '-'*30

    # print len(usr_plain_list), len(sys_plain_list)
    # # Test merge_dialog
    # for i in range(len(dialog[0])):
    #     history = dialog[0][i]
    #     print i
    #     for j in range(len(history)):
    #         # print '-----Number of Turns: %d' % j
    #         print history[j]

    # # Test convert_2D_str2id
    # for i in range(len(dialog[1])):
    #     print i
    #     print dialog[1][i]
    #     print dialog_id[1][i]

    # # Test read_sys_template
    # for k in template_dict.keys():
    #     print k, template_dict[k]
    # print len(template_dict.keys())
    sys_list = flatten_2D(sys_list)
    f = open('data/template_label.txt', 'w')
    for i in range(len(label_list)):
        f.write('%s\n' % (label_list[i]))
    f.close()