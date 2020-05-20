import re, jieba, numpy as np
from utils.textprocessing import load_embedding_matrix, process_sentence
from collections import Counter
from sklearn.decomposition import PCA
from utils.data_loader import load_data


class SentEmbGenerator:
    def __init__(self, a=0.001):
        """

        :param a: The factor that used to calculate weighted sentence embedding
        """
        embedding_matrix, word_index_dict = load_embedding_matrix()
        self.embedding_matrix = embedding_matrix
        self.word_index_dict = word_index_dict
        jieba.load_userdict('data/user_dict.txt')
        self.embedding_size = len(self.embedding_matrix[0])
        self.a = a

    def split_into_sentences(self, text):
        """

        :param text: Input text
        :return: A list of sentences
        """
        pattern = r'\||\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|。|、|；|‘|’|【|】|·|！| |…|（|）'
        result_list = re.split(pattern, text)
        return [line for line in result_list if line is not None and len(line) > 0]

    def calculate_sentence_vector_helper(self, sentence):
        """

        :param sentence: Input sentence
        :return: The weighted average sentence vector
        """
        words = sentence.split()
        word_freq_dict = Counter(words)
        vs = np.zeros(self.embedding_size)
        for word in words:
            if word in self.word_index_dict:
                word_embedding = self.embedding_matrix[int(self.word_index_dict[word])]
            else:
                word_embedding = self.embedding_matrix[int(self.word_index_dict['<unknown>'])]

            word_freq = word_freq_dict[word] / len(words)
            a_value = self.a / (self.a + word_freq)
            vs = np.add(vs, np.multiply(a_value, word_embedding))

        vs = np.divide(vs, len(words))
        return vs

    def sentences_to_vecs(self, sentence_list):
        """

        :param sentence_list: Input sentence list
        :return: A list of sentence vectors
        """
        vec_list = []
        for sentence in sentence_list:
            vec_list.append(self.calculate_sentence_vector_helper(sentence))

        pca = PCA()
        pca.fit(np.array(vec_list))
        u = pca.components_[0]
        u = np.multiply(u, np.transpose(u))

        if len(u) < self.embedding_size:
            for i in range(self.embedding_size - len(u)):
                u = np.append(u, 0)

        sentence_vecs = []
        for vec in vec_list:
            sub = np.multiply(u, vec)
            sentence_vecs.append(np.subtract(vec, sub))

        return sentence_vecs

    def generate_sentence_vectors(self, text):
        """

        :param text: Input text
        :return: A list of sentences in this text and sentence vectors of these sentences
        """
        sentence_list = self.split_into_sentences(text)
        sentences = []
        trans_sentences = []
        for sentence in sentence_list:
            trans_sentence = process_sentence(sentence)
            trans_sentence = trans_sentence.replace('<start>', '').replace('<end>', '').strip()
            if trans_sentence is not None and len(trans_sentence) > 0:
                sentences.append(sentence)
                trans_sentences.append(trans_sentence)
        return sentences, self.sentences_to_vecs(trans_sentences)


if __name__ == '__main__':
    train_X, train_Y, test_X = load_data(transform_data=False)
    sent_emd_generator = SentEmbGenerator()
    sentences, sent_vecs = sent_emd_generator.generate_sentence_vectors(train_X[0])
    print(sent_vecs)


