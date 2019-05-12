import logging
import mxnet as mx
from sknlp.classifier import TextRCNNClassifier, TextCNNClassifier, TextRNNClassifier
from sklearn.metrics import precision_recall_fscore_support
from sknlp.tagger import TextRNNTagger
from sknlp.elmo import AdaptiveSoftmax


def multi2single(intents):

    order = ('查快递', '转人工', '运费', '取件时间', '下单', '否认', '确认',
             '打招呼', '其他')
    mapping = {
        '查快递': ['其他催单', '催促', '催单', '查快递', '时效', '抱怨'],
        '转人工': ['转人工'],
        '运费': ['运费'],
        '取件时间': ['取件时间'],
        '下单': ['下单'],
        '否认': ['否认'],
        '确认': ['确认'],
        '打招呼': ['打招呼'],
        '其他': ['其他']}

    for single in order:
        if any(origin in intents for origin in mapping[single]):
            return single
    else:
        return 'nonsense'


logging.basicConfig(level=logging.DEBUG)

if __name__ == '__main__':
    clf = TextRNNClassifier(3, is_multilabel=True, segmenter=None)
    X, y = (['123abi', '123你好', '123你好abc', '你好', 'abc', '123'],
            ['n|a', 'n|w', 'n|a|w', 'w', 'a', 'n'])
    clf.fit(X, y, valid_X=X, valid_y=y, n_epochs=30, checkpoint='temp/test')
    print(clf.predict(['abc123123', '1你好23']))

    # clf = TextRCNNClassifier(3, is_multilabel=False, segmenter=None)
    # X, y = (['123a', '1你好啊', '2你好啊c', '你好', 'abc', '123', 'ab1c', '12b3', 'ab你c', '1啊23'],
    #         ['n', 'w', 'w', 'w', 'a', 'n', 'a', 'n', 'a', 'n'])
    # clf.fit(X, y, valid_X=X, valid_y=y, n_epochs=12)
    # print(clf.predict(['你好a啊', '你123']))

    # train, test = load_intent_dataset()
    # print(len(train), len(test))
    # # clf = TextCNNClassifier(14, True,
    # #                         num_filters=(25, 50, 75, 100, 125, 150),
    # #                         ngram_filter_sizes=(1, 2, 3, 4, 5, 6))
    # # clf.fit(train_dataset=train, valid_dataset=test, n_epochs=60, checkpoint='dev/word')
    # for i in range(20, 21):
    #     print(f'----------------------------------------model {i:02}')
    #     clf = TextCNNClassifier.load_model('test.tar.gz')
    #     scores = clf.score(dataset=test)
    #     for l, p, r, f, _ in zip(
    #             clf.idx2labels(list(range(clf._num_classes))), *scores):
    #         print(f'label: {l} {f * 100}({p * 100}, {r * 100})')
    #     print('-------------------------------')

    #     predictions = [multi2single(p) for p in clf.predict(dataset=test)]
    #     y = [multi2single(intent)
    #          for intent in clf._decode_label(clf._debinarize([labels for _, _, labels in test]))]
    #     labels = list(set(y))
    #     scores = precision_recall_fscore_support(y, predictions, labels=labels)
    #     for l, p, r, f, _ in zip(labels, *scores):
    #         print(f'label: {l} {f * 100}({p * 100}, {r * 100})')
    #     print(precision_recall_fscore_support(y, predictions, average='micro'))
    #     # clf.save_model('test')
    # # train, test = load_msra_dataset(segmenter=None)
    # # print(len(train), len(test), len(train.label2idx))
    # # # print(train.label2idx)
    # # # clf = TextRNNTagger(len(train.label2idx))
    # # # clf.fit(train_dataset=train, valid_dataset=test, n_epochs=60, checkpoint='dev/tagger')
    # # clf = TextRNNTagger.load_model('dev/tagger.tar.gz')
    # # print(clf.score(dataset=test))
    # x = mx.nd.array([[0.6939, 0.2245, 0.8520, 0.9945, 0.0832, 0.0112, 0.0476, 0.0417],
    #                  [0.8796, 0.8644, 0.4868, 0.5303, 0.1786, 0.3198, 0.3256, 0.6437],
    #                  [0.7293, 0.0287, 0.2014, 0.3848, 0.0433, 0.4146, 0.4891, 0.1255]])
    # head_w = mx.nd.array([[0.2810, 0.0114, -0.2531, 0.0009, -0.2719, -0.0525, 0.0934, -0.1241],
    #                       [0.1130, -0.1605, 0.0147, -0.2976, -0.3403, 0.3490, 0.1385, 0.3341],
    #                       [-0.0894, 0.2479, -0.2980, -0.0052, 0.3377, -0.1223, 0.2446, -0.3370]])
    # tail_proj = mx.nd.array([[-0.2761, 0.3084, -0.2970, 0.1652, 0.1107, -0.3243, -0.2168, -0.1898],
    #                          [0.2112, -0.0661, 0.1959, -0.0734, -0.0302, 0.1943, -0.1861, 0.2701]])
    # tail_w = mx.nd.array([[-0.1995, 0.0196],
    #                       [-0.2380, -0.3977],
    #                       [-0.0473, 0.0400],
    #                       [0.6211, 0.3060],
    #                       [-0.5989, -0.6056],
    #                       [0.1567, 0.5375],
    #                       [-0.2101, -0.0704],
    #                       [-0.3419, 0.3128]])
    # target = mx.nd.array([0, 1, 7])
    # soft = AdaptiveSoftmax(8, 10, [2, ])
    # soft.initialize()
    # soft.head_layer.weight.set_data(head_w)
    # soft.tail_layers[0][0].weight.set_data(tail_proj)
    # soft.tail_layers[0][1].weight.set_data(tail_w)
    # print(soft(x, target))
