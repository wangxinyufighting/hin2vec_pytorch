import numpy as np
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score, average_precision_score)
def geiEmbedding(n_file):
    node_embeddings = {}
    with open(n_file, 'r', encoding='utf8') as f:
        for line in f.readlines()[1:]:
            temp = line.strip().split()
            name = temp[0]
            embedding = np.array([float(i) for i in temp[1:]])
            node_embeddings[name] = embedding
    return node_embeddings

def getTestData(file):
    true_edges = []
    neg_edges = []
    with open(file, 'r', encoding='utf8') as f:
        for line in f.readlines():
            temp = line.strip().split()
            node1 = temp[1]
            node2 = temp[2]
            flag = temp[3]
            if flag == '0':
                true_edges.append((node1, node2))
            else:
                neg_edges.append((node1, node2))

    return true_edges, neg_edges

def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass

def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)

    return roc_auc_score(y_true, y_scores), \
           f1_score(y_true, y_pred, average='micro'), \
           f1_score(y_true, y_pred, average='macro'),\
           auc(rs, ps), \
           average_precision_score(y_true, y_scores)


true_edges, neg_edges = getTestData(r'test.txt')
# node_embeddings = geiEmbedding('merge1/node_vec_merge_1_5_5_100_100_5_5_True_1_64_200.txt')
node_embeddings = geiEmbedding('node_vec_og_p100_5_5_55_200_5_True_1_64_200.txt')
auc_roc, f1_mi, f1_ma, auc_pr, ap = evaluate(node_embeddings, true_edges, neg_edges)
print('Overall ROC-AUC:', auc_roc)
print('Overall PR-AUC', auc_pr)
print('Overall F1-micro:', f1_mi)
print('Overall F1-macro:', f1_ma)
print('Overall mAP:', ap)





