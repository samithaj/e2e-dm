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
    cuisine_path = 'data/cuisine.txt'
    location_path = 'data/location.txt'
    price_path = 'data/price.txt'
    post_code_path = 'data/post_code.txt'
    phone_path = 'data/phone.txt'
    address_path = 'data/address.txt'
    with open(kb_path) as f:
        lines = f.readlines()
    f_cuisine = open(cuisine_path, 'w')
    f_location = open(location_path, 'w')
    f_price = open(price_path, 'w')
    f_post_code = open(post_code_path, 'w')
    f_phone = open(phone_path, 'w')
    f_address = open(address_path, 'w')
    f_values = {'R_cuisine': f_cuisine,
                'R_location': f_location,
                'R_price': f_price,
                'R_post_code': f_post_code,
                'R_address': f_address,
                'R_phone': f_phone}
    names = []
    values = {'R_cuisine': [],
              'R_location': [],
              'R_price': [],
              'R_post_code': [],
              'R_address': [],
              'R_phone': []}
    num_line = len(lines)
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
    f_post_code.close()
    f_phone.close()
    f_address.close()
    return names, values


# Save the vocabulary list into a plain txt file
def save_vocab(data_path, kb_path):
    vocab_path = 'data/all_vocab.txt'
    names, values, val2attr, entities = dl.read_kb_value(kb_path)
    index_list, usr_list, sys_list = dl.read_dialog(data_path)
    num_turn = len(usr_list)
    vocab_dict = {}
    # print val2attr
    for i in range(num_turn):
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
            print item
            f.write('%s %d\n' % item)


# check if a sentence is api_call
def get_api_call(s):
    s = s.strip('\n').split(' ')
    hotword = s[0]
    if hotword == 'api_call':
        return s[1:]
    else:
        return []


if __name__ == '__main__':
    all_path = 'data/dialog-babi-task6-dstc2-all.txt'
    kb_path = 'data/dialog-babi-task6-dstc2-kb.txt'
    # save_kb_value(kb_path)
    save_vocab(all_path, kb_path)

