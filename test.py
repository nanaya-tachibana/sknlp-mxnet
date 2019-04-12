from sknlp.classifier import TextCNNClassifier, TextRNNClassifier
from sknlp.dataset import load_waimai_dataset, load_intent_dataset
from sklearn.metrics import precision_recall_fscore_support


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


if __name__ == '__main__':
    # clf = TextCNNClassifier(3, True, segmenter=list)
    # X, y = (['123abi', '123你好', '123你好abc', '你好', 'abc', '123'],
    #         ['n|a', 'n|w', 'n|a|w', 'w', 'a', 'n'])
    # clf.fit(X, y, valid_X=X, valid_y=y, n_epochs=12)
    # print(clf.predict(['abc123123', '1你好23']))

    # clf = TextCNNClassifier(3, False, segmenter=list)
    # X, y = (['123a', '1你好啊', '2你好啊c', '你好', 'abc', '123', 'ab1c', '12b3', 'ab你c', '1啊23'],
    #         ['n', 'w', 'w', 'w', 'a', 'n', 'a', 'n', 'a', 'n'])
    # clf.fit(X, y, valid_X=X, valid_y=y, n_epochs=12)
    # print(clf.predict(['你好a啊', '你123']))

    train, test = load_intent_dataset()
    print(len(train), len(test))
    # clf = TextCNNClassifier(14, True,
    #                         num_filters=(25, 50, 75, 100, 125, 150),
    #                         ngram_filter_sizes=(1, 2, 3, 4, 5, 6))
    # clf.fit(train_dataset=train, valid_dataset=test, n_epochs=60, checkpoint='dev/word')
    for i in range(20, 21):
        print(f'----------------------------------------model {i:02}')
        clf = TextCNNClassifier.load_model('dev/word-vocab.json', 'dev/word-meta.json',
                                           (f'dev/word-0-00{i:02}-params', ))
        # scores = clf.score(dataset=test)
        # for l, p, r, f, _ in zip(
        #         clf.idx2labels(list(range(clf._num_classes))), *scores):
        #     print(f'label: {l} {f * 100}({p * 100}, {r * 100})')
        # print('-------------------------------')

        # predictions = [multi2single(p) for p in clf.predict(dataset=test)]
        # y = [multi2single(intent)
        #      for intent in clf._decode_label(clf._debinarize([labels for _, _, labels in test]))]
        # labels = list(set(y))
        # scores = precision_recall_fscore_support(y, predictions, labels=labels)
        # for l, p, r, f, _ in zip(labels, *scores):
        #     print(f'label: {l} {f * 100}({p * 100}, {r * 100})')
        # print(precision_recall_fscore_support(y, predictions, average='micro'))
        print(clf.predict(dataset=test))
