import pandas as pd, jieba, re, numpy as np, os
from gensim.models.word2vec import Word2Vec, LineSentence
from gensim.models.fasttext import FastText


def clean_sentence(sentence):
    '''

    :param sentence: original sentence
    :return: the sentence removed special characters
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ''


def separate_words_in_sentence(sentence):
    """

    :param sentence: original sentence
    :return: the sentence whose words have been separated
    """
    result = []
    tokens = sentence.split('|')
    for token in tokens:
        result.append(' '.join(list(jieba.cut(token))))
    return ' | '.join(result)


def remove_stop_words(sentence):
    """

    :param sentence: original sentence
    :return: the sentence removed stop words
    """
    words = sentence.split()
    with open('drive/kaikeba/Abstract/data/stopwords.txt', 'r+', encoding='utf-8') as stop_words_file:
        content = stop_words_file.read()
        stop_words = set(content.splitlines())
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


def process_sentence(sentence):
    """

    :param sentence: original sentence
    :return: processed sentence
    """
    sentence = clean_sentence(sentence)
    sentence = separate_words_in_sentence(sentence)
    sentence = remove_stop_words(sentence)
    sentence = '<start> ' + sentence + ' <end>'
    return sentence


def transform_original_dataframe(df):
    """

    :param df: original dataframe
    :return: processed dataframe
    """
    for col_name in ['Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(lambda x: process_sentence(x))

    if 'Report' in df.columns:
        df['Report'] = df['Report'].apply(lambda x: process_sentence(x))
    return df


def build_model(data_path, sg=0, hs=0, negative=5, size=100, min_count=5, workers=8, window=5, model_type='word2vec'):
    if model_type == 'word2vec':
        model = Word2Vec(LineSentence(data_path), sg=sg, workers=workers, min_count=min_count, size=size, hs=hs, negative=negative, window=window)
    elif model_type == 'fasttext':
        model = FastText(LineSentence(data_path), sg=sg, workers=workers, min_count=min_count, size=size, hs=hs, negative=negative, window=window)
    else:
        raise Exception('Model type is not supported.')
    vectors = model.wv.vectors
    vocabulary = model.wv.index2word
    with open('index_to_vectors.txt', 'w+', encoding='utf-8') as vectors_file:
        content = '\n'.join(
            ['{}\t{}'.format(i + 1, ','.join([str(item) for item in vector])) for i, vector in enumerate(vectors)])
        content = '{}\t{}\n'.format(0, ','.join(['0'] * size)) + content
        vectors_file.write(content)
    with open('data/word_to_vectors.txt', 'w+', encoding='utf-8') as vectors_file:
        content = '\n'.join(['{}\t{}'.format(vocabulary[i], ','.join([str(item) for item in vector])) for i, vector in
                             enumerate(vectors)])
        content = '{}\t{}\n'.format('<pad>', ','.join(['0'] * size)) + content
        vectors_file.write(content)
    with open('data/vocabulary.txt', 'w+', encoding='utf-8') as vocab_file:
        content = '\n'.join(['{}\t{}'.format(i + 1, str(vocab)) for i, vocab in enumerate(vocabulary)])
        content = '{}\t{}\n'.format(0, '<pad>') + content
        vocab_file.write(content)
    model_path = 'word2vec.model'
    model.save(model_path)
    return model_path


def load_embedding_matrix():
    with open('drive/kaikeba/Abstract/data/word_to_vectors.txt', 'r+', encoding='utf-8') as vectors_file:
        content = vectors_file.read()
        lines = content.splitlines()
        embedding_size = len(lines[0].split('\t')[1].split(','))
        word_vector_dict = {line.split('\t')[0]: line.split('\t')[1].split(',') for line in lines}
    with open('drive/kaikeba/Abstract/data/vocabulary.txt', 'r+', encoding='utf-8') as vocab_file:
        content = vocab_file.read()
        lines = content.splitlines()
        vocab_size = len(lines)
        vocab_index_dict = {line.split('\t')[1]: int(line.split('\t')[0]) for line in lines}
    embedding_matrix = np.zeros((vocab_size, embedding_size))
    for word in vocab_index_dict.keys():
        embedding_matrix[int(vocab_index_dict[word])] = word_vector_dict[word]
    return embedding_matrix, vocab_index_dict


if __name__ == '__main__':
    jieba.load_userdict('user_dict.txt')

    train_df = pd.read_csv('data/AutoMaster_TrainSet.csv', encoding='utf-8')
    test_df = pd.read_csv('data/AutoMaster_TestSet.csv', encoding='utf-8')
    trans_train_df = transform_original_dataframe(train_df)
    trans_test_df = transform_original_dataframe(test_df)

    merged_df = pd.concat(
        [trans_train_df['Question'], trans_train_df['Dialogue'], trans_train_df['Report'], trans_test_df['Question'],
         trans_test_df['Dialogue']], axis=0)
    if os.path.exists('data/vocabulary.txt'):
        with open('data/vocabulary.txt', 'r', encoding='utf-8') as vocab_file:
            content = vocab_file.read()
            lines = content.splitlines()
            vocab_index_dict = {line.split('\t')[1]: int(line.split('\t')[0]) for line in lines}
        merged_df = merged_df.apply(lambda x: ' '.join([item if item in vocab_index_dict else '<unknown>' for item in x.split()]))
    merged_df.to_csv('merged_w2v_training_data.csv', encoding='utf-8', index=False)

    model_path = build_model('data/merged_w2v_training_data.csv', model_type='fasttext')
    embedding_matrix, vocab_index_dict = load_embedding_matrix()
