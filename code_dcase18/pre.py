import os
import csv
import shutil

csv_file = csv.reader(open('F:\ESC-50-master\meta\esc50.csv','r'))
# ['filename', 'fold', 'target', 'category', 'esc10', 'src_file', 'take']
# ['1-100032-A-0.wav', '1', '0', 'dog', 'True', '100032', 'A']
audioPath = 'F:\ESC-50-master\\audio'
prePath = 'F:\ESC-50-master\ESC-50'
print (csv_file)

csv_new_file = [col for col in csv_file]
for au in csv_new_file:
    if not au[1].isdigit():
        continue
    # print (au)
    if int(au[2]) < 10 :
        au[2] = '00' + au[2]
    elif int(au[2]) < 100 :
        au[2] = '0' + au[2]
    path = prePath +'\\' + au[2]+' - '+ au[3];
    # print (path)
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print (path + '创建成功')
    else :
        print (path + '目录已存在')
    target_file = audioPath + '\\' + au[0]
    new_file = path + '\\' + au[0]
    print (target_file)
    if os.path.isfile(target_file):
        print ('找到文件')
        shutil.copyfile(target_file , new_file)
