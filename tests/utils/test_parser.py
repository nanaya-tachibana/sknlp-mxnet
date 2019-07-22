from sknlp.utils.parser import parse_tagged_text


class TestParseTaggedText():

    def test_empty_input(self):
        text = ''
        tags = []
        assert parse_tagged_text(text, tags) == {}

    def test_normal_input(self):
        text = '哦北京市会话吴中区上海市'
        tags = [
            'O', 'B-C', 'I-C', 'I-C', 'O', 'O',
            'B-D', 'I-D', 'I-D', 'B-C', 'I-C', 'I-C'
        ]
        assert parse_tagged_text(text, tags) == {
            'C': ['北京市', '上海市'],
            'D': ['吴中区']
        }
