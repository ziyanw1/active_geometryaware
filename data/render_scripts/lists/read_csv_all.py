import os
import sys
import csv

FILE_NAME = 'all.csv'
CATEGORY = '03001627'

folder_name = CATEGORY+'_lists'

if not os.path.exists(folder_name):
    os.mkdir(folder_name)

if __name__ == '__main__':
    
    with open(FILE_NAME) as f:
        f_reader = csv.reader(f)
        i = 0
        f_train = open(os.path.join('./', folder_name, 'train_idx.txt'), 'w')
        f_val = open(os.path.join('./', folder_name, 'val_idx.txt'), 'w')
        f_test = open(os.path.join('./', folder_name, 'test_idx.txt'), 'w')

        for row in f_reader:
            #i += 1
            #print row[1]
            #print row[3]
            #print row[4]
            #if i > 5:
            #    break

            if row[1] == CATEGORY:
                if row[4] == 'train':
                    f_train.write(row[3]+'\n')
                elif row[4] == 'val':
                    f_val.write(row[3]+'\n')
                elif row[4] == 'test':
                    f_test.write(row[3]+'\n')

                print 'writing {} to {} list'.format(row[3], row[4])

