
with open('text_exports.txt', 'w') as f:
    f.write('hello')

data = [1,2,3,4]
f = open('text_exports.txt', 'w')
f.write('something')

with open('export_test.txt', 'w') as f:
    for i in data:
        f.write(str(i) + '\n')
