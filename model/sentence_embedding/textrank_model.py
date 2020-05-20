import networkx as nx, numpy as np
from model.sentence_embedding.sentence_embedding_generator import SentEmbGenerator
from sklearn.metrics.pairwise import cosine_similarity
from utils.data_loader import load_data


class TextRankModel:
    def __init__(self, a=0.001, selected_num=3):
        """

        :param a: The factor that used to calculate weighted sentence embedding
        :param selected_num: The total number of sentences we will selected to construct our summary
        """
        self.sent_emb_generator = SentEmbGenerator(a)
        self.selected_num = selected_num

    def predict_single_item(self, text):
        """

        :param text: Input text
        :return: Summary for this text
        """
        sentences, sent_vecs = self.sent_emb_generator.generate_sentence_vectors(text)
        sim_mat = np.zeros([len(sentences), len(sentences)])
        for i in range(len(sentences)):
            for j in range(len(sentences)):
                if i != j:
                    sim_mat[i][j] = cosine_similarity([sent_vecs[i]], [sent_vecs[j]])[0, 0]

        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph)

        ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

        result = ''
        for i in range(min(len(sentences), self.selected_num)):
            result += (ranked_sentences[i][1] + '。')

        return result

    def predict(self, text_list):
        """

        :param text_list: Input text list
        :return: Summaries for these texts
        """
        result = []
        for text in text_list:
            result.append(self.predict_single_item(text))
        return result


if __name__ == '__main__':
    train_X, train_Y, test_X = load_data(transform_data=False)
    model = TextRankModel()
    result = model.predict(train_X[:2])
    print(result)
