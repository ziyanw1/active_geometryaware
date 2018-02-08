import os
import sys
import json


list_name = 'render_ids.txt'

if __name__ == '__main__':
    
    ids = []
    with open(list_name, 'r') as f:
        ids = f.readlines()

    #print ids
    for id_name, i in zip(ids, range(len(ids))):
        id_name = id_name[:-1]
        ids[i] = id_name
    print ids
    test_dict = {}
    car_dict = {}
    car_dict['name'] = 'car,auto,automobile,machine,motorcar'
    car_dict['train'] = []
    car_dict['val'] = []
    car_dict['test'] = ids
    test_dict['02958343'] = car_dict

    print test_dict

    with open('test_split_1.json', 'w') as f:
        json.dump(test_dict, f)
