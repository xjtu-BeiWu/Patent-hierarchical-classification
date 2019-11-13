import numpy as np
import tflearn
import tensorflow
from sklearn.metrics import accuracy_score, f1_score, hamming_loss


# test_labels_trans = np.array([[1, 0, 1], [1, 0, 0], [1, 1, 1]])
# test_predict_trans = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 1]])
# accuracy = accuracy_score(test_labels_trans, test_predict_trans)
# f1_score_macro = f1_score(test_labels_trans, test_predict_trans, average='macro')
# f1_score_micro = f1_score(test_labels_trans, test_predict_trans, average='micro')
# f1_score_weighted = f1_score(test_labels_trans, test_predict_trans, average='weighted')
# hamming = hamming_loss(test_labels_trans, test_predict_trans)
# print("accuracy:", accuracy)
#
# print("f1_score_macro:", f1_score_macro)
# print("f1_score_micro:", f1_score_micro)
# print("f1_score_weighted:", f1_score_weighted)
# print("hamming_loss:", hamming)


#
# with tensorflow.Session() as sess:
#     test = np.array([[0.01, 0.08, 0.91], [1, 0, 0], [0.65, 0.30, 0.05]])
#     network = tflearn.sigmoid(test)
#     out = output(sess.run(network))
#     # tflearn.objectives.binary_crossentropy()
#     # output = tflearn
#     # logits = tensorflow.nn.xw_plus_b(network, W, b, name="logits")
#     print(sess.run(network))
#     print(out)
