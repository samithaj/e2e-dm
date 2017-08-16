# coding:utf-8

'''
  This script is for data pre-processing.
  Dataset introduction:
    Number of restaurants: 113
    Number of cuisine: 24
    Number of price: 3 (expensive, moderate, cheap)
    Number of locations: 5 (south, west, centre, north, east)

  Created on Aug 7, 2017
  Author: qihu@mobvoi.com
'''

import data_loader as dl


# Save the attributes list in knowledge base to plain txt
def save_kb_value(kb_path):
    cuisine_path = 'babi5/cuisine.txt'
    location_path = 'babi5/location.txt'
    price_path = 'babi5/price.txt'
    post_code_path = 'babi5/number.txt'
    phone_path = 'babi5/phone.txt'
    address_path = 'babi5/address.txt'
    rating_path = 'babi5/rating.txt'
    with open(kb_path) as f:
        lines = f.readlines()
    f_cuisine = open(cuisine_path, 'w')
    f_location = open(location_path, 'w')
    f_price = open(price_path, 'w')
    f_number = open(post_code_path, 'w')
    f_phone = open(phone_path, 'w')
    f_address = open(address_path, 'w')
    f_rating = open(rating_path, 'w')
    f_values = {'R_cuisine': f_cuisine,
                'R_location': f_location,
                'R_price': f_price,
                'R_number': f_number,
                'R_address': f_address,
                'R_phone': f_phone,
                'R_rating': f_rating}
    names = []
    values = {'R_cuisine': [],
              'R_location': [],
              'R_price': [],
              'R_number': [],
              'R_address': [],
              'R_phone': [],
              'R_rating': []}
    num_line = len(lines)
    print num_line
    for i in range(num_line):
        line = lines[i].strip('\n').split(' ')
        attribute = line[2]
        value = line[3]
        if value not in values[attribute]:
            values[attribute].append(value)
    for attr in values.keys():
        for v in values[attr]:
            f_values[attr].write('%s\n' % v)

    f_cuisine.close()
    f_location.close()
    f_price.close()
    f_number.close()
    f_phone.close()
    f_address.close()
    f_rating.close()
    return names, values


# Save the vocabulary list into a plain txt file
def save_vocab(data_path, kb_path):
    vocab_path = 'babi5/data/all_vocab.txt'
    names, values, val2attr, entities = dl.read_kb_value(kb_path)
    usr_list, sys_list, api_list = dl.read_dialog(data_path)

    vocab_dict = {}
    # print val2attr
    usr_list = dl.flatten_2D(usr_list)
    sys_list = dl.flatten_2D(sys_list)
    num_turn = len(usr_list)
    print len(usr_list), len(sys_list)
    for i in range(num_turn):
        print i
        sentence = ('%s %s' % (usr_list[i], sys_list[i])).split(' ')
        for word in sentence:
            if word in val2attr.keys():
                word = val2attr[word]
            if word in names:
                word = 'R_name'
            if word in vocab_dict.keys():
                vocab_dict[word] = vocab_dict[word]+1
            else:
                vocab_dict[word] = 1
    items = vocab_dict.items()
    items = sorted(items, lambda x, y: cmp(x[1], y[1]), reverse=True)
    with open(vocab_path, 'w') as f:
        for item in items:
            print item[0]
            f.write('%s %d\n' % item)


# check if a sentence is api_call
def get_api_call(s):
    s = s.strip('\n').split(' ')
    hotword = s[0]
    if hotword == 'api_call':
        return s[1:]
    else:
        return []


def save_diff_sys_resp(data_path, kb_path):
    save_path = 'data/template/sys_resp.txt'
    sys_resp_list = []
    names, values, val2attr, entities = dl.read_kb_value(kb_path)
    # print names
    with open(data_path) as f:
        lines = f.readlines()
    f = open(save_path, 'w')
    num_line = len(lines)
    for i in range(num_line):
        line = lines[i].strip('\n').split('\t')
        if len(line) < 2:
            continue
        sys_resp = line[1]
        sys_word = sys_resp.split(' ')
        num_word = len(sys_word)
        for j in range(num_word):
            word = sys_word[j]
            if word in val2attr:
                sys_word[j] = val2attr[word]
            if word in names:
                sys_word[j] = 'R_name'
        s = ' '.join(sys_word)
        if s in sys_resp_list:
            continue
        sys_resp_list.append(s)
    sys_resp_list.sort()
    for s in sys_resp_list:
        f.write('%s\n' % s)
    f.close()
    return 0


def sort_sentence(data_path):
    save_path = 'data/template/sys_resp_templete_2_sorted.txt'
    s_list = []
    with open(data_path) as f:
        lines = f.readlines()
    f = open(save_path, 'w')
    num_line = len(lines)
    for i in range(num_line):
        s = lines[i]
        s_list.append(s)
    s_list.sort()
    for s in s_list:
        f.write(s)
    f.close()


# Find out all system response with "ask" (to reduce conflict in restaurant name "ask")
def save_ask(data_path):
    ask_list = []
    save_path = 'data/ask_sentence'
    with open(data_path) as f:
        lines = f.readlines()
    f = open(save_path, 'w')
    num_line = len(lines)
    for i in range(num_line):
        line = lines[i].split('\t')
        if len(line) < 2:
            continue
        line = line[1]
        words = line.strip('\n').split(' ')
        if 'ask' in words and line not in ask_list:
            ask_list.append(line)
    for s in ask_list:
        f.write(s)
    f.close()


if __name__ == '__main__':
    all_path = 'babi5/data/dialog-babi-task5-full-dialogs-all.txt'
    trn_path = 'babi5/data/dialog-babi-task5-full-dialogs-trn.txt'
    dev_path = 'babi5/data/dialog-babi-task5-full-dialogs-dev.txt'
    tst_path = 'babi5/data/dialog-babi-task5-full-dialogs-tst.txt'
    kb_path = 'babi5/data/dialog-babi-kb-all.txt'
    # save_kb_value(kb_path)
    save_vocab(all_path, kb_path)
    # save_diff_sys_resp(all_path, kb_path)
    # sort_sentence('data/template/sys_resp_template_2.txt')
    # save_ask(tst_path)