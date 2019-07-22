from collections import defaultdict


def parse_tagged_text(text, tags):
    results = defaultdict(list)
    current_tag = None
    start, i = 0, 0
    for i, tag in enumerate(tags):
        if tag[0] == 'I' and current_tag == tag[2:]:
            continue

        if i != start:
            results[current_tag].append(text[start:i])

        if tag == 'O':
            start = i + 1
            current_tag = None
        elif tag[0] == 'B' or tag[0] == 'I':
            start = i
            current_tag = tag[2:]

    if i != start and current_tag is not None:
        results[current_tag].append(text[start:i + 1])
    return results
