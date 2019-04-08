from classifier import TextCNNClassifier


if __name__ == '__main__':
    clf = TextCNNClassifier(3, True, segmenter=list)
    X, y = (['123abi', '123你好', '123你好abc', '你好', 'abc', '123'],
            ['n|a', 'n|w', 'n|a|w', 'w', 'a', 'n'])
    clf.fit(X, y, n_epochs=10)
    print(clf.predict(['abc123123', '1你好23']))

    clf = TextCNNClassifier(3, False, segmenter=list)
    X, y = (['123a', '1你好啊', '2你好啊c', '你好', 'abc', '123', 'ab1c', '12b3', 'ab你c', '1啊23'],
            ['n', 'w', 'w', 'w', 'a', 'n', 'a', 'n', 'a', 'n'])
    clf.fit(X, y, n_epochs=10)
    print(clf.predict(['你好a啊', '你123']))
