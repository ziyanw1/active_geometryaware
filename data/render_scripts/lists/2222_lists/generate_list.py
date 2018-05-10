import os
import sys

total_num = 107

train_name = 'train_idx.txt'
val_name = 'val_idx.txt'
test_name = 'test_idx.txt'
all_name = 'all_idx.txt'

train_list = [str(i)+'\n' for i in range(75)]
val_list = [str(i+75)+'\n' for i in range(10)] 
test_list = [str(i+85)+'\n' for i in range(22)]
all_list = [str(i)+'\n' for i in range(107)]

with open(train_name, 'w') as f:
    f.writelines(train_list)

with open(val_name, 'w') as f:
    f.writelines(val_list)

with open(test_name, 'w') as f:
    f.writelines(test_list)

with open(all_name, 'w') as f:
    f.writelines(all_list)

print train_list[:10]
