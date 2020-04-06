import codecs
import os
import re

from data_utils import create_dico, create_mapping, zero_digits
from data_utils import iob2, iob_iobes, get_seg_features


def load_sentences(path, lower, zeros):
    """
    Load sentences. A line must contain at least a word and its tag.
    Sentences are separated by empty lines.
    returns：sentences. 2-D, [[['伊', 'B-LOC'], ['拉', 'I-LOC'], ['克', 'I-LOC'] ...], ...]
    """
    sentences = []  # 2-D
    sentence = []
    with open(path, encoding='UTF-8') as f:
        lines = f.readlines()
    for line in lines:   # eg: '你   O'
        line = zero_digits(line.rstrip()) if zeros else line.rstrip()
        if not line:
            if len(sentence) > 0:
                # 'DOCSTART' 没有出现过
                if 'DOCSTART' not in sentence[0][0]:
                    sentences.append(sentence)
                sentence = []
        else:
            if line[0] == " ":  # 未出现过
                line = "$" + line[1:]
                word = line.split()
                word[0] = " "
            else:
                word = line.split()
            assert len(word) == 2
            sentence.append(word)
    if len(sentence) > 0:
        if 'DOCSTART' not in sentence[0][0]:
            sentences.append(sentence)
    return sentences


def update_tag_scheme(sentences, tag_scheme):
    """
    Check and update sentences tagging scheme to IOB2.
    Only IOB1 and IOB2 schemes are accepted.
    return: sentences. IOB -> IOBES. [[['伊', 'B-LOC'], ['拉', 'I-LOC'], ['克', 'E-LOC'] ...], ...]
    """
    for i, s in enumerate(sentences):
        tags = [w[-1] for w in s]
        # Check that tags are given in the IOB format
        if not iob2(tags):
            s_str = '\n'.join(' '.join(w) for w in s)
            raise Exception('Sentences should be given in IOB format! ' +
                            'Please check sentence %i:\n%s' % (i, s_str))
        if tag_scheme == 'iob':
            # If format was IOB1, we convert to IOB2
            for word, new_tag in zip(s, tags):
                word[-1] = new_tag
        elif tag_scheme == 'iobes':
            new_tags = iob_iobes(tags)
            for word, new_tag in zip(s, new_tags):
                word[-1] = new_tag
        else:
            raise Exception('Unknown tagging scheme!')


def char_mapping(sentences, lower):
    """
    Create a dictionary and a mapping of words, sorted by frequency.
    returns：dico, {'迈': 151, '向': 2212, '充': 556, '满': 719, ...}
             char_to_id, {'<PAD>': 0, '<UNK>': 1, ',': 2, '的': 3, '0': 4, ...}
             id_to_char, {0: '<PAD>', 1: '<UNK>', 2: ',', 3: '的', 4: '0', ...}
    """
    chars = [[x[0].lower() if lower else x[0] for x in s] for s in sentences]

    dico = create_dico(chars)  # {char: frequency, ...}
    dico["<PAD>"] = 10000001
    dico['<UNK>'] = 10000000
    char_to_id, id_to_char = create_mapping(dico)
    return dico, char_to_id, id_to_char


def tag_mapping(sentences):
    """
    Create a dictionary and a mapping of tags, sorted by frequency.
    returns: dico, {'O': 1608733, 'B-TIME': 15975, ...}
             tag_to_id, {'O': 0, 'I-ORG': 1, 'I-TIME': 2, ...}
             id_to_tag, {0: 'O', 1: 'I-ORG', 2: 'I-TIME', ...}
    """
    f = open('tag_to_id.txt', 'w', encoding='utf8')
    f1 = open('id_to_tag.txt', 'w', encoding='utf8')
    tags = []  # 2-D
    for s in sentences:
        ts = []
        for char in s:
            tag = char[-1]
            ts.append(tag)
        tags.append(ts)
    
    #tags1 = [[char[-1] for char in s] for s in sentences]
    dico = create_dico(tags)  # {'O': 1608733, 'B-TIME': 15975, ...}
    tag_to_id, id_to_tag = create_mapping(dico)
    #print("Found %i unique named entity tags" % len(dico))
    for k, v in tag_to_id.items():
        f.write(k+":"+str(v)+"\n")
    for k, v in id_to_tag.items():
        f1.write(str(k) + ":" + str(v) + "\n")
    return dico, tag_to_id, id_to_tag

def augment_with_pretrained(dictionary, ext_emb_path, chars):
    """
    Augment the dictionary with words that have a pretrained embedding.
    If `words` is None, we add every word that has a pretrained embedding
    to the dictionary, otherwise, we only add the words that are given by
    `words` (typically the words in the development and test sets.)
    """
    # print('Loading pretrained embeddings from %s...' % ext_emb_path)
    assert os.path.isfile(ext_emb_path)

    # Load pretrained embeddings from file
    pretrained = set([
        line.rstrip().split()[0].strip()
        for line in codecs.open(ext_emb_path, 'r', 'utf-8')
        if len(ext_emb_path) > 0
    ])  # set, {'冷', '防’, ...}

    # We either add every word in the pretrained file,
    # or only words given in the `words` list to which we can assign a pretrained embedding
    if chars is None:
        for char in pretrained:
            if char not in dictionary:
                dictionary[char] = 0
    else:
        for char in chars:
            if any(x in pretrained for x in [char, char.lower(), re.sub('\d', '0', char.lower())]) and char not in dictionary:
                dictionary[char] = 0

    word_to_id, id_to_word = create_mapping(dictionary)
    return dictionary, word_to_id, id_to_word


def prepare_dataset(sentences, char_to_id, tag_to_id, lower=False, train=True):
    """
    Prepare the dataset. Return a list of lists of dictionaries containing:
        - word indexes
        - word char indexes
        - tag indexes
    returns: string, ['迈', '向', '充', '满', '希', '忘', '的', ...]
             chars,  [1260, 181,  663,  551,  565, 436,   3,  ...]
             segs,   [1,     3,    1,    2,    2,   3,    0, ... ]
             tags,   [0,     0,    0,    0,    0,   0,    0, ... ]  # 分词特性
    """
    none_index = tag_to_id["O"]
    def f(x):
        return x.lower() if lower else x
    data = []
    for s in sentences:
        string = [w[0] for w in s]
        chars = [char_to_id[f(w) if f(w) in char_to_id else '<UNK>'] for w in string]
        segs = get_seg_features("".join(string))
        if train:
            tags = [tag_to_id[w[-1]] for w in s]
        else:
            tags = [none_index for _ in chars]
        data.append([string, chars, segs, tags])

    return data


def save_maps(save_path, *params):
    """
    Save mappings and invert mappings
    """
    pass
    # with codecs.open(save_path, "w", encoding="utf8") as f:
    #     pickle.dump(params, f)


def load_maps(save_path):
    """
    Load mappings from the file
    """
    pass
    # with codecs.open(save_path, "r", encoding="utf8") as f:
    #     pickle.load(save_path, f)


# if __name__ == '__main__':
#     lower = False
#     zeros = True
#
#     train_path = 'C:\\DATASET\\NER\\train.txt'
#     # train_path = 'D:\\PROJECTS\\DATA\\NER\\train.txt'
#     train_sentences = load_sentences(train_path, lower, zeros)
#
#     tag_scheme = 'iobes'
#     update_tag_scheme(train_sentences, tag_scheme)
#     dico_train = char_mapping(train_sentences, lower)[0]
#
#     emb_path = 'C:\\DATASET\\NER\\vec.txt'
#     test_path = 'C:\\DATASET\\NER\\test.txt'
#     # emb_path = 'D:\\PROJECTS\\DATA\\NER\\vec.txt'
#     # test_path = 'D:\\PROJECTS\\DATA\\NER\\test.txt'
#     test_sentences = load_sentences(test_path, lower, zeros)
#     import itertools
#
#     test_chars = list(itertools.chain.from_iterable([[w[0] for w in s] for s in test_sentences]))
#     dico_chars, char_to_id, id_to_char = augment_with_pretrained(dico_train, emb_path, test_chars)
#
#     dico, tag_to_id, id_to_tag = tag_mapping(train_sentences)  # train + dev + test
#     train_data = prepare_dataset(train_sentences, char_to_id, tag_to_id, lower)
#
#     from data_utils import BatchManager
#     batch_size = 20
#     train_manager = BatchManager(train_data, batch_size)






