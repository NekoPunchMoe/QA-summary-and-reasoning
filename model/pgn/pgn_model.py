import time, tensorflow as tf, math, random
from model.pgn.layers import Encoder, Decoder, Attention, Pointer
from utils.textprocessing import load_embedding_matrix
from utils.data_loader import load_data

class AttentionModel(tf.keras.Model):
    def __init__(self, rnn_type='GRU', use_coverage=True, cov_loss_weight=0.5, bidirectional=False, score_type='additive-concat', 
                 max_length_input=100, max_length_output=100, min_length_output = 3,
                 embedding_matrix=None, word_index_dict=None, batch_size=64,
                 encoder_units=128, attention_units=128, decoder_units=128, epochs=2, learning_rate=0.15, decay_rate=0.98):
        """

        :param rnn_type: The type of recurrent neural network we used
        :param bidirectional: Whether our neural network is bidirectional
        :param score_type: he type of attention score we used
        :param max_length_input: The max length of input strings
        :param max_length_output: The max length of output strings
        :param embedding_matrix: Pretrained embedding matrix
        :param word_index_dict: Dictionary contains word and index pair information
        :param batch_size: Batch size
        :param encoder_units: The number of units in encoder layer
        :param attention_units: The number of units in attention layer
        :param decoder_units: The number of units in decoder layer
        :param epochs: The number of epochs
        """
        
        super(AttentionModel, self).__init__()
        
        # initiate params
        self.rnn_type = rnn_type
        self.use_coverage = use_coverage
        self.cov_loss_weight = cov_loss_weight
        self.bidirectional = bidirectional
        self.score_type = score_type
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output
        self.min_length_output = min_length_output

        self.embedding_matrix = embedding_matrix
        self.word_index_dict = word_index_dict
        self.index_word_dict = {self.word_index_dict[item]: item for item in self.word_index_dict}
        self.pad_index = self.word_index_dict['<pad>']

        self.batch_size = batch_size
        self.encoder_units = encoder_units
        self.attention_units = attention_units
        self.decoder_units = decoder_units
        self.epochs = epochs
        self.vocab_size = len(self.embedding_matrix)
        self.embedding_dim = len(self.embedding_matrix[0])

        # build layers
        self.encoder = Encoder(self.vocab_size, self.embedding_dim, self.encoder_units, self.batch_size, rnn_type=self.rnn_type,
                          embedding_matrix=self.embedding_matrix, bidirectional=self.bidirectional)
        self.attention = Attention(self.attention_units, score_type=self.score_type, mask_index=self.pad_index)
        self.decoder = Decoder(self.vocab_size, self.embedding_dim, self.decoder_units, self.batch_size, rnn_type=self.rnn_type,
                          embedding_matrix=self.embedding_matrix)
        self.pointer = Pointer()

        # initiate optimizer and loss function
        self.optimizer = tf.keras.optimizers.Adagrad(learning_rate,
                                                     initial_accumulator_value=0.1,
                                                     clipnorm=2.0)
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='none')
        
        # save model
        self.checkpoint_dir = 'drive/kaikeba/Abstract/data/checkpoints/training_checkpoints1'
        self.checkpoint = tf.train.Checkpoint(PGN=self)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=5)
        self.decay_rate = decay_rate

    # loss calculation related
    def loss_function(self, real, pred, pad_mask):
        """

        :param real: True item from input
        :param pred: Predicted item from model
        :return: loss value
        """

        loss = 0
        for t in range(real.shape[1]):
            loss_ = self.loss_object(real[:, t], pred[:, t, :])
            mask = tf.cast(pad_mask[:, t], dtype=loss_.dtype)
            loss_ *= mask
            loss_ = tf.reduce_mean(loss_, axis=0)  # batch-wise
            loss += loss_

        return loss / real.shape[1]

    def coverage_loss(self, attn_dists, coverages, padding_mask):
        '''

        :param attn_dists: A list of attention weights
        :param coverages:  A list of coverage scores
        :param padding_mask: A list to determine whether the element at current position is a padding
        :return: coverage loss
        '''
        # cover_losses = []
        # # transfer attn_dists coverages to [max_len_y, batch_sz, max_len_x]
        # attn_dists = tf.convert_to_tensor(attn_dists)
        # coverages = tf.squeeze(coverages, -1)

        # assert attn_dists.shape == coverages.shape
        # for t in range(attn_dists.shape[0]):
        #     cover_loss_ = tf.reduce_sum(tf.minimum(attn_dists[t, :, :], coverages[t, :, :]), axis=-1)  # max_len_x wise
        #     print(cover_loss_)
        #     cover_losses.append(cover_loss_)
        # print(padding_mask)
        coverage = tf.zeros_like(attn_dists[0])  # shape (batch_size, attn_length). Initial coverage is zero.
        cover_losses = []  # Coverage loss per decoder timestep. Will be list length max_dec_steps containing shape (batch_size).
        for a in attn_dists:
            covloss = tf.reduce_sum(tf.minimum(a, coverage), [1])  # calculate the coverage loss for this step
            cover_losses.append(covloss)
            coverage += a  # update the coverage vector
        # change from[max_len_y, batch_sz] to [batch_sz, max_len_y]
        cover_losses = tf.stack(cover_losses, 1)
        # cover_loss_ [batch_sz, max_len_y]
        mask = tf.cast(padding_mask, dtype=covloss.dtype)
        cover_losses *= mask
        loss = tf.reduce_mean(tf.reduce_mean(cover_losses, axis=0))  # mean loss of each time step and then sum up
        # tf.print('coverage loss(batch sum):', loss)
        return loss

    # data transform related
    def tokenize_one_sentence(self, sentence, extended_vocab=None):
        """

        :param sentence: sentence to be tokenized
        :param extended_vocab:
        If extended_vocab is None, we will store oovs in a list and return it. And we will generated extended tokenized sentence based on it.
        Otherwise we will use extended_vocab to generate extended tokenized sentence
        :return: a tuple contains:
        1. tokenized sentence
        2. extended tokenized sentence
        3. a list of extended vocabularies
        """
        tokenized_sentence = []
        extended_tokenized_sentence = []

        # only used when extended_vocab is None
        oov_list = []

        sentence = sentence.split()
        for word in sentence:
            if word in self.word_index_dict.keys():
                tokenized_sentence.append(self.word_index_dict[word])
                extended_tokenized_sentence.append(self.word_index_dict[word])
            else:
                tokenized_sentence.append(self.word_index_dict['<unknown>'])
                if extended_vocab is None:
                    if word not in oov_list:
                        oov_list.append(word)
                    oov_num = oov_list.index(word)
                    extended_tokenized_sentence.append(len(self.word_index_dict) + oov_num)
                else:
                    if word in extended_vocab:
                        oov_num = extended_vocab.index(word)
                        extended_tokenized_sentence.append(len(self.word_index_dict) + oov_num)
                    else:
                        extended_tokenized_sentence.append(self.word_index_dict['<unknown>'])

        return tokenized_sentence, extended_tokenized_sentence, oov_list

    def tokenize_data(self, x, y):
        """

        :param x: Inputs we used to train
        :param y: Expected outputs for given inputs
        :return: A dictionary contains 6 elements:
        1. a list of tokenized input sentence
        2. a list of extended tokenized input sentence
        3. a list of tokenized output sentence
        4. a list of extended tokenized output sentence
        6. a list of the length of extended vocabularies
        6. a list of extended vocabularies
        """
        result = {
            'input': [],
            'input_extended': [],
            'output': [],
            'output_extended': [],
            'oov_len': [],
            'oov_list': []
        }
        if y is not None:
            for input, output in zip(x, y):
                tokenized_x, extended_x, oov_list = self.tokenize_one_sentence(input)
                tokenized_y, extended_y, _ = self.tokenize_one_sentence(output, extended_vocab=oov_list)
                result['input'].append(tokenized_x)
                result['input_extended'].append(extended_x)
                result['output'].append(tokenized_y)
                result['output_extended'].append(extended_y)
                result['oov_len'].append(len(oov_list)),
                result['oov_list'].append(oov_list)
        else:
            for input in x:
                tokenized_x, extended_x, oov_list = self.tokenize_one_sentence(input)
                result['input'].append(tokenized_x)
                result['input_extended'].append(extended_x)
                result['oov_len'].append(len(oov_list)),
                result['oov_list'].append(oov_list)
        return result

    def generate_dataset(self, x, y):
        tokenized_result = self.tokenize_data(x, y)

        tokenized_result['input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['input'], maxlen=self.max_length_input, padding='post', value=self.pad_index)
        tokenized_result['input_extended'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['input_extended'], maxlen=self.max_length_input, padding='post', value=self.pad_index)
        tokenized_result['output'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['output'], maxlen=self.max_length_output, padding='post', value=self.pad_index)
        tokenized_result['output_extended'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['output_extended'], maxlen=self.max_length_output, padding='post', value=self.pad_index)
        dataset = tf.data.Dataset.from_tensor_slices((tokenized_result['input'], tokenized_result['input_extended'], tokenized_result['output'], tokenized_result['output_extended'], tokenized_result['oov_len'])).shuffle(self.batch_size)
        dataset = dataset.batch(self.batch_size, drop_remainder=True)
        return dataset

    # combine attention distribution and vocab distribution
    def calculate_final_distribution(self, input_extended, vocab_dists, atten_dists, pgen_list, oov_len):
        '''

        :param input_extended: The input of encoder layer with extended vocabulary id. (batch_size, input_len)
        :param vocab_dists: The distribution generated by original attention model. (output_len, batch_size, vocab_size)
        :param atten_dists: A list of attention weights. (output_len, batch_size, input_len)
        :param pgen_list: A list that contains pgen scores. (output_len, batch_size, 1)
        :param oov_len: Max length of oov list in current batch
        :return: Final distributions.
        '''
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(pgen_list, vocab_dists)]  # shape (output_len, batch_size, vocab_size)
        atten_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(pgen_list, atten_dists)]  # shape (output_len, batch_size, input_len)
        batch_size = vocab_dists[0].shape[0]

        extended_vsize = len(self.word_index_dict) + oov_len
        extra_zeros = tf.zeros((batch_size, oov_len))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) for dist in vocab_dists]  # shape (output_len, batch_size, extended_vsize)

        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)  # shape (batch_size, 1)
        attn_len = tf.shape(input_extended)[1]  # number of states we attend over
        batch_nums = tf.tile(batch_nums, [1, attn_len])  # shape (batch_size, attn_len)
        indices = tf.stack((batch_nums, input_extended), axis=2)  # shape (batch_size, enc_t, 2)
        shape = [batch_size, extended_vsize]

        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) for copy_dist in atten_dists]  # shape (output_len, batch_size, extended_vsize)

        final_dists = [vocab_dist + copy_dist for (vocab_dist, copy_dist) in
                       zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists  # shape (output_len, batch_size, extended_vsize)

    # training related
    def train_one_step(self, input, target, enc_hidden, input_extended, output_extended, oov_len):
        '''

        :param input: The input of encoder layer
        :param target: Expected output for the given input
        :param enc_hidden: initial hidden state to input encoder layer
        :param input_extended: The input of encoder layer with extended vocabulary id
        :param output_extended: Expected output for the given input with extended vocabulary id
        :param oov_len: Max length of oov list in current batch
        :return: batch loss for this step
        '''
        loss = 0

        with tf.GradientTape() as tape:
            encoder_pad_mask = tf.math.logical_not(tf.math.equal(input, self.pad_index))
            decoder_pad_mask = tf.math.logical_not(tf.math.equal(target, self.pad_index))[: , 1:]
            enc_output, enc_hidden = self.encoder(input, enc_hidden)
            dec_hidden = enc_hidden
            dec_input = tf.expand_dims([self.word_index_dict['<start>']] * self.batch_size, 1)

            pred_list = []
            attention_list = []
            pgen_list = []
            cov_list = []
            prev_coverage = None
            threshhold = 1

            # Teacher forcing - feeding the target as the next input
            for t in range(1, target.shape[1]):
                context_vector, attention_weights, prev_coverage = self.attention(dec_hidden, enc_output, encoder_pad_mask, self.use_coverage, prev_coverage)
                predictions, dec_hidden = self.decoder(dec_input, dec_hidden, context_vector)
                pgen = self.pointer(context_vector, dec_hidden, dec_input)
                pred_list.append(predictions)
                attention_list.append(attention_weights)
                pgen_list.append(pgen)
                cov_list.append(prev_coverage)

                # using scheduled sampling
                cur_samp = random.uniform(0, 1)
                # if cur_samp < threshhold:
                dec_input = tf.expand_dims(target[:, t], 1)
                # else:
                    # dec_input = tf.expand_dims(tf.math.argmax(predictions, axis=-1), 1)

                threshhold *= self.decay_rate

            final_dists = self.calculate_final_distribution(input_extended, pred_list, attention_list, pgen_list, oov_len)
            final_dists = tf.stack(final_dists, 1)

            log_loss = self.loss_function(output_extended[:, 1:], final_dists, decoder_pad_mask)
            batch_loss = log_loss

            if self.use_coverage:
                batch_loss += self.coverage_loss(attention_list, cov_list, decoder_pad_mask)

            variables = self.encoder.trainable_variables + self.decoder.trainable_variables + self.attention.trainable_variables + self.pointer.trainable_variables
            # print('origin:')
            # print(variables)

            gradients = tape.gradient(batch_loss, variables)
            # print(gradients)

            self.optimizer.apply_gradients(zip(gradients, variables))
            # print('changed:')
            # print(variables)

            return batch_loss, log_loss

    def fit(self, x, y):
        """

        :param x: Inputs we used to train
        :param y: Expected outputs for given inputs
        :return:
        """
        steps_per_epoch = len(x) // self.batch_size
        dataset = self.generate_dataset(x, y)

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            total_log_loss = 0

            for (batch, (inp, inp_ext, targ, targ_ext, oov_len)) in enumerate(dataset.take(steps_per_epoch)):
                batch_start = time.time()
                max_oov_len = tf.math.reduce_max(oov_len)
                batch_loss, log_loss = self.train_one_step(inp, targ, enc_hidden, inp_ext, targ_ext, max_oov_len)
                total_loss += batch_loss
                total_log_loss += log_loss

                if batch % 1 == 0:
                    if not self.use_coverage:
                        print('Epoch {} Batch {} Loss {:.4f} Time {} sec'.format(epoch + 1,
                                                                    batch,
                                                                    batch_loss.numpy(),
                                                                    time.time() - batch_start))
                    else:
                        print('Epoch {} Batch {} Loss {:.4f} Log Loss {:.4f} Coverage Loss {:.4f} Time {} sec'.format(epoch + 1,
                                                                                                          batch,
                                                                                                          batch_loss.numpy(),
                                                                                                          log_loss.numpy(),
                                                                                                          batch_loss.numpy() - log_loss.numpy(),
                                                                                                          time.time() - batch_start))
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                ckpt_save_path = self.checkpoint_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))
            print('Epoch {} Loss {:.4f} log loss {:.4f} cov loss {:.4f}'.format(epoch + 1,
                                                total_loss / steps_per_epoch,
                                                total_log_loss / steps_per_epoch,
                                                (total_loss - total_log_loss) / steps_per_epoch))
            print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # beam search related
    def get_top_k_for_one_step(self, enc_output, dec_input, dec_hidden, beam_size, input_extended, prev_coverage, max_oov_len):
        encoder_pad_mask = tf.math.logical_not(tf.math.equal(input_extended, self.pad_index))
        context_vector, attention_weights, prev_coverage = self.attention(dec_hidden, enc_output, encoder_pad_mask, self.use_coverage, prev_coverage)
        predictions, dec_hidden = self.decoder(dec_input, dec_hidden, context_vector)
        pgen = self.pointer(context_vector, dec_hidden, dec_input)

        final_dists = self.calculate_final_distribution(input_extended, [predictions], [attention_weights], [pgen], max_oov_len)
        final_dists = tf.stack(final_dists, 1)

        top_k_probs, top_k_ids = tf.nn.top_k(tf.squeeze(final_dists), k=beam_size)
        top_k_log_probs = tf.math.log(top_k_probs)
        if len(enc_output) == 1:
          top_k_ids = tf.expand_dims(top_k_ids, 0)
          top_k_log_probs = tf.expand_dims(top_k_log_probs, 0)
        return dec_hidden, prev_coverage, top_k_log_probs, top_k_ids

    def beam_predict_single_item(self, enc_output, dec_hidden, beam_size, alpha, return_best, input_extended, oov_list, oov_len):
        steps = 0
        candidates = [(0, 0, [self.word_index_dict['<start>']], dec_hidden[0])]
        results = []

        prev_coverage = None
        while steps < self.max_length_output and len(results) < beam_size and len(candidates) > 0:
            last_tokens = [item[2][-1] if item[2][-1] in self.index_word_dict else self.word_index_dict['<unknown>'] for item in candidates]
            dec_hidden_list = [item[3] for item in candidates]
            temp_dec_input = tf.expand_dims(last_tokens, 1)
            temp_dec_hidden = tf.convert_to_tensor(dec_hidden_list)
            temp_input_extended = tf.tile(input_extended, [len(temp_dec_input), 1])
            temp_enc_output = tf.tile(enc_output, [len(temp_dec_input), 1, 1])
            max_oov_len = tf.math.reduce_max(oov_len)

            dec_hidden, prev_coverage, top_k_log_probs, top_k_ids = self.get_top_k_for_one_step(temp_enc_output, temp_dec_input, temp_dec_hidden, beam_size, temp_input_extended, prev_coverage, max_oov_len)

            cur_candidates = []

            for i in range(len(candidates)):
                cur_item = candidates[i]
                for j in range(beam_size):
                    cur_token_list = cur_item[2] + [top_k_ids[i, j].numpy()]
                    cur_tot_log_prob = cur_item[1] + top_k_log_probs[i, j].numpy()
                    cur_avg_log_prob = cur_tot_log_prob / math.pow(len(cur_token_list), alpha)
                    cur_dec_hidden = dec_hidden[i]
                    cur_candidates.append([cur_avg_log_prob, cur_tot_log_prob, cur_token_list, cur_dec_hidden])

            candidates = []
            sorted_candidates = sorted(cur_candidates, key=lambda c: c[0], reverse=True)

            for candidate in sorted_candidates:
                if candidate[2][-1] == self.word_index_dict['<end>']:
                    if len(candidate[2]) >= self.min_length_output:
                        results.append(candidate)
                else:
                    candidates.append(candidate)

                if len(candidates) == beam_size or len(results) == beam_size:
                    break

            steps += 1

        if len(results) == 0:
            results = candidates

        if len(results) == 0:
          raise Exception('There is no suitable answer.')

        sorted_result = sorted(results, key=lambda r: r[0], reverse=True)
        if not return_best:
            return sorted_result
        else:
            best_result = results[0]
            return ' '.join([self.index_word_dict[index] if index in self.index_word_dict else oov_list[index - len(self.index_word_dict)] for index in best_result[2]])

    def beam_predict(self, x, beam_size=1, alpha=0, return_best=True):
        """

        :param x: Inputs we want to predict
        :param beam_size: The number of beam size
        :param alpha: The factor to determine how many punishment we will apply to short sentence.
        :param return_best: Whether to return the best result or all result in beam bucket.
        :return:
        1. If return_best is true, then this function will return a sentence whose average log probability sum is largest.
        2. If return_best is false, then this function will return a list of tuples which contains best beam_size results.
           And the tuple contains three elements: average log probability sum, total log probability sum, a list of word index.
        """
        tokenized_result = self.tokenize_data(x, y=None)
        tokenized_result['input'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['input'], maxlen=self.max_length_input, padding='post', value=self.pad_index)
        tokenized_result['input_extended'] = tf.keras.preprocessing.sequence.pad_sequences(tokenized_result['input_extended'], maxlen=self.max_length_input, padding='post', value=self.pad_index)

        enc_hidden = self.encoder.initialize_hidden_state(batch_size=len(tokenized_result['input']))
        enc_output, enc_hidden = self.encoder(tokenized_result['input'], enc_hidden)
        dec_hidden = enc_hidden

        result = []

        for i in range(len(tokenized_result['input'])):
            cur_enc_output = enc_output[i: i + 1]
            cur_dec_hidden = dec_hidden[i: i + 1]
            int_ext = tokenized_result['input_extended'][i: i + 1]
            oov_len = tokenized_result['oov_len'][i: i + 1]
            oov_list =tokenized_result['oov_list'][i]

            cur_result = self.beam_predict_single_item(cur_enc_output, cur_dec_hidden, beam_size, alpha, return_best, int_ext, oov_list, oov_len)
            result.append(cur_result)

        return result


if __name__ == '__main__':
    embedding_matrix, word_index_dict = load_embedding_matrix()
    train_x, train_y, test_x = load_data()
    x = train_x[:600]
    y = train_y[:600]
    model = AttentionModel(embedding_matrix=embedding_matrix, word_index_dict=word_index_dict)
    model.fit(x, y)

    model.beam_predict(test_x[:2], beam_size=3)
    model.beam_predict(test_x[:2], beam_size=5)
    model.beam_predict(test_x[:2], beam_size=5, return_best=False)
    model.beam_predict(test_x[:2], beam_size=5, alpha=0.5, return_best=False)
