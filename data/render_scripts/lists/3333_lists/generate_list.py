import os
import sys

total_num = 332

train_name = 'train_idx.txt'
val_name = 'val_idx.txt'
test_name = 'test_idx.txt'

train_list = [str(i)+'\n' for i in range(232)]
val_list = [str(i+232)+'\n' for i in range(33)] 
test_list = [str(i+265)+'\n' for i in range(67)]

with open(train_name, 'w') as f:
    f.writelines(train_list)

with open(val_name, 'w') as f:
    f.writelines(val_list)

with open(test_name, 'w') as f:
    f.writelines(test_list)

print train_list[:10]
