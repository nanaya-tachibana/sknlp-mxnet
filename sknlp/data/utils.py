import logging
import re

import mxnet as mx
import re


def strip_whitespace(s):
    return re.sub('[\s\t\n]+', '', s)
