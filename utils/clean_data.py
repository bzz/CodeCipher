import json
import sys

args = sys.argv

with open(args[1]) as f:
    data = f.readlines()
data = [json.loads(line) for line in data]
for iter in range(len(data)):

    if '```' in data[iter]['completion']:
        idx = data[iter]['completion'].find('```')
        data[iter]['completion'] = data[iter]['completion'][:idx]
    if '\n\n\n# Test' in data[iter]['completion']:
        idx = data[iter]['completion'].find('\n\n\n# Test')
        data[iter]['completion'] = data[iter]['completion'][:idx]

    func_name_left = data[iter]['completion'].find('def ')
    if func_name_left > 0 and data[iter]['completion'][func_name_left - 1] == ' ':
        func_name_left = data[iter]['completion'].find('def ')
    
    func_name_left = func_name_left + len('def ')
    func_name_right = data[iter]['completion'].find('(', func_name_left)
    func_name = data[iter]['completion'][func_name_left:func_name_right]
    data[iter]['entry_point'] = func_name.split(':')[0]

with open(args[2], 'w') as f:
    for iter in range(len(data)):
        f.write(json.dumps(data[iter]) + '\n')