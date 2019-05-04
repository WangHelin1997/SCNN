import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from util import *

true = np.load('label_all.npy')
pred = np.load('pred_all.npy')
labelx = []
for i in range(49):
    labelx.append(i+1)
labelx.append(0)
labelx = np.array(labelx)
# print(label)
# print(true)
# print(pred)
label= []
for i in range(len(labelx)):
    label.append(id_to_lb(labelx[i]))
tick_marks = np.array(range(len(label))) + 0.5

# label = [0,1,2,3,4,5,6,7,8,9]
# label = np.array(label)
# label = ['airport','shopping_mall','metro_station', 'street_pedestrian','public_square', 'street_traffic','tram', 'bus','metro', 'park']
# tick_marks = np.array(range(len(label))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(label)))
    plt.xticks(xlocations, label, rotation=90)
    plt.yticks(xlocations, label)
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

cm = confusion_matrix(true, pred)
# cm = [[218,118,17,52,6,6,4,0,0,0],
#       [49,271,40,66,14,0,0,0,0,1],
#       [19,42,281,11,11,4,20,14,33,0],
#       [10,20,17,313,48,8,8,3,1,1],
#       [22,10,23,61,166,54,14,6,2,29],
#       [1,0,7,16,20,348,0,5,1,4],
#       [0,4,22,9,3,1,274,55,66,2],
#       [2,3,30,0,0,3,109,228,39,1],
#       [3,5,91,0,1,0,34,22,277,0],
#       [1,0,17,7,13,26,8,0,0,314]]
cm = np.array(cm)
print(cm.shape)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print(cm_normalized)
plt.figure(figsize=(12, 8), dpi=120)

ind_array = np.arange(len(label))
x, y = np.meshgrid(ind_array, ind_array)

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm_normalized[y_val][x_val]
    if c > 0.01:
        plt.text(x_val, y_val, "%0.2f" % (c,), color='red', fontsize=7, va='center', ha='center')
# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-')
plt.gcf().subplots_adjust(bottom=0.15)
# plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix on DCASE Task1')
plot_confusion_matrix(cm_normalized, title='Normalized confusion matrix on ESC-50')
# show confusion matrix
plt.savefig('confusion_matrix.png', format='png')
plt.show()

