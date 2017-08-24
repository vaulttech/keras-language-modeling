from __future__ import print_function

from abc import abstractmethod

from keras.engine import Input
from keras.layers import merge, Embedding, Dropout, Convolution1D, Lambda, LSTM, Dense
from keras import backend as K
from keras.models import Model
from keras.layers.advanced_activations import LeakyReLU
from keras import regularizers, initializers
from keras import losses


import numpy as np


class LanguageModel:
    def __init__(self, config):
        #self.question = Input(shape=(config['question_len'],), dtype='int32', name='question_base')
        #self.answer_good = Input(shape=(config['answer_len'],), dtype='int32', name='answer_good_base')
        #self.answer_bad = Input(shape=(config['answer_len'],), dtype='int32', name='answer_bad_base')

        self.question = Input(shape=(config['question_len'],), dtype='float32', name='question_base')
        self.answer_good = Input(shape=(config['answer_len'],), dtype='float32', name='answer_good_base')
        self.answer_bad = Input(shape=(config['answer_len'],), dtype='float32', name='answer_bad_base')

        self.config = config
        self.params = config.get('similarity', dict())

        # initialize a bunch of variables that will be set later
        self._models = None
        self._similarities = None
        self._answer = None
        self._qa_model = None

        self.training_model = None
        self.prediction_model = None

    def get_answer(self):
        if self._answer is None:
            self._answer = Input(shape=(self.config['answer_len'],), dtype='float32', name='answer')
        return self._answer

    @abstractmethod
    def build(self):
        return

    def get_similarity(self, similarity=None):
        ''' Specify similarity in configuration under 'similarity' -> 'mode'
        If a parameter is needed for the model, specify it in 'similarity'

        Example configuration:

        config = {
            ... other parameters ...
            'similarity': {
                'mode': 'gesd',
                'gamma': 1,
                'c': 1,
            }
        }

        cosine: dot(a, b) / sqrt(dot(a, a) * dot(b, b))
        polynomial: (gamma * dot(a, b) + c) ^ d
        sigmoid: tanh(gamma * dot(a, b) + c)
        rbf: exp(-gamma * l2_norm(a-b) ^ 2)
        euclidean: 1 / (1 + l2_norm(a - b))
        exponential: exp(-gamma * l2_norm(a - b))
        gesd: euclidean * sigmoid
        aesd: (euclidean + sigmoid) / 2
        '''

        params = self.params
        if similarity is None:
            similarity = params['mode']

        dot = lambda a, b: K.batch_dot(a, b, axes=1)
        l2_norm = lambda a, b: K.sqrt(K.sum(K.square(a - b), axis=1, keepdims=True))

        if similarity == 'cosine':
            return lambda x: dot(x[0], x[1]) / K.maximum(K.sqrt(dot(x[0], x[0]) * dot(x[1], x[1])), K.epsilon())
        elif similarity == 'polynomial':
            return lambda x: (params['gamma'] * dot(x[0], x[1]) + params['c']) ** params['d']
        elif similarity == 'sigmoid':
            return lambda x: K.tanh(params['gamma'] * dot(x[0], x[1]) + params['c'])
        elif similarity == 'rbf':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]) ** 2)
        elif similarity == 'euclidean':
            return lambda x: 1 / (1 + l2_norm(x[0], x[1]))
        elif similarity == 'exponential':
            return lambda x: K.exp(-1 * params['gamma'] * l2_norm(x[0], x[1]))
        elif similarity == 'gesd':
            euclidean = lambda x: 1 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 1 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) * sigmoid(x)
        elif similarity == 'aesd':
            euclidean = lambda x: 0.5 / (1 + l2_norm(x[0], x[1]))
            sigmoid = lambda x: 0.5 / (1 + K.exp(-1 * params['gamma'] * (dot(x[0], x[1]) + params['c'])))
            return lambda x: euclidean(x) + sigmoid(x)
        else:
            raise Exception('Invalid similarity: {}'.format(similarity))

    def get_qa_model(self):
        if self._models is None:
            self._models = self.build()

        if self._qa_model is None:
            question_output, answer_output = self._models
            dropout = Dropout(self.params.get('dropout', 0.2))
            similarity = self.get_similarity()
            if (self.config['loss_type'] == 'all'):
                qa_model =  merge([dropout(question_output), dropout(answer_output)],
                            mode=similarity,
                            output_shape=lambda _: (None, 1))
            else: # elif (self.config['loss_type'] == 'none'):
                cosine = merge([dropout(question_output), dropout(answer_output)],
                            mode=self.get_similarity('cosine'),
                            output_shape=lambda _: (None, 1))
                polynomial = merge([dropout(question_output), dropout(answer_output)],
                            mode=self.get_similarity('polynomial'),
                            output_shape=lambda _: (None, 1))
                sigmoid = merge([dropout(question_output), dropout(answer_output)],
                            mode=self.get_similarity('sigmoid'),
                            output_shape=lambda _: (None, 1))
                rbf = merge([dropout(question_output), dropout(answer_output)],
                                mode=self.get_similarity('rbf'),
                                output_shape=lambda _: (None, 1))
                euclidean = merge([dropout(question_output), dropout(answer_output)],
                                mode=self.get_similarity('euclidean'),
                                output_shape=lambda _: (None, 1))
                exponential = merge([dropout(question_output), dropout(answer_output)],
                                mode=self.get_similarity('exponential'),
                                output_shape=lambda _: (None, 1))
                gesd = merge([dropout(question_output), dropout(answer_output)],
                                mode=self.get_similarity('gesd'),
                                output_shape=lambda _: (None, 1))
                aesd = merge([dropout(question_output), dropout(answer_output)],
                                mode=self.get_similarity('aesd'),
                                output_shape=lambda _: (None, 1))
                qa_model = K.concatenate([cosine, polynomial, sigmoid, rbf,
                                          euclidean, exponential, gesd, aesd])

            self._qa_model = Model(input=[self.question, self.get_answer()], output=qa_model, name='qa_model')
            self.meta_model = Model(input=[self.question, self.get_answer()],
                    output=[qa_model, question_output, answer_output], name='meta_model')

        return self._qa_model

    def compile(self, optimizer, **kwargs):
        qa_model = self.get_qa_model()

        good_similarity = qa_model([self.question, self.answer_good])
        bad_similarity = qa_model([self.question, self.answer_bad])

        loss = merge([good_similarity, bad_similarity],
                     mode=lambda x: K.relu(self.config['margin'] - x[0] + x[1]),
                     output_shape=lambda x: x[0])

        if (self.config['loss_type'] == 'all'):
            final_similarity = Dense(1, activation='sigmoid')
            final_good_similarity = final_similarity(good_similarity)
            final_bad_similarity = final_similarity(bad_similarity)
            loss = losses.mean_squared_error(final_good_similarity, 1) + \
                    losses.mean_squared_error(final_bad_similarity, 0)
        elif (self.config['loss_type'] == 'reconstruction'):
            pass
        # else: #(self.config['loss_type'] == 'loss'):
        #     pass


        self.prediction_model = Model(input=[self.question, self.answer_good], output=good_similarity, name='prediction_model')
        self.prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

        self.meta_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

        self.training_model = Model(
                input=[self.question, self.answer_good, self.answer_bad],
                output=loss,
                name='training_model')
        self.training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self.training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=(x[0].shape[0],)) # doesn't get used
        return self.training_model.fit(x, y, **kwargs)

    def meta_predict(self, x):
        assert self.meta_model is not None and isinstance(self.meta_model, Model)
        return self.meta_model.predict_on_batch(x)

    def predict(self, x):
        assert self.prediction_model is not None and isinstance(self.prediction_model, Model)
        return self.prediction_model.predict_on_batch(x)

    def save_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model before saving weights'
        self.prediction_model.save_weights(file_name, **kwargs)

    def load_weights(self, file_name, **kwargs):
        assert self.prediction_model is not None, 'Must compile the model loading weights'
        self.prediction_model.load_weights(file_name, **kwargs)

class MLPModel(LanguageModel):
    def build_with_size(self, size):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        if (self.config['weight_decay']):
            fclayer = Dense(size, activation='linear',
                    kernel_regularizer=regularizers.l2(0.1))
        else:
            fclayer = Dense(size, activation='linear')

        if (self.config['use_leaky_relu']):
            leaky_relu = LeakyReLU(alpha=0.3)
            question_embedding = leaky_relu(fclayer(question))
            answer_embedding = leaky_relu(fclayer(answer))
        else:
            question_embedding = fclayer(question)
            answer_embedding = fclayer(answer)

        # maxpooling
        #maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        #maxpool.supports_masking = True
        #question_pool = maxpool(question_embedding)
        #answer_pool = maxpool(answer_embedding)

        return question_embedding, answer_embedding

class MLPModel_100(MLPModel):
    def build(self):
        return self.build_with_size(100)

class MLPModel_200(MLPModel):
    def build(self):
        return self.build_with_size(200)

class MLPModel_300(MLPModel):
    def build(self):
        return self.build_with_size(300)

class MLPModel_400(MLPModel):
    def build(self):
        return self.build_with_size(400)

class MLPModel_500(MLPModel):
    def build(self):
        return self.build_with_size(500)

class DoubleMLPModel(LanguageModel):
    def build_with_size(self, size):
        question = self.question
        answer = self.get_answer()

        if (self.config['weight_decay']):
            fclayer = Dense(size, activation='linear',
                            kernel_regularizer=regularizers.l2(0.01))
        else:
            fclayer = Dense(size, activation='linear')

        if (self.config['use_leaky_relu']):
            leaky_relu = LeakyReLU(alpha=0.3)
            q = leaky_relu(fclayer(question))
            a = leaky_relu(fclayer(answer))
        else:
            q = fclayer(question)
            a = fclayer(answer)

        if (self.config['weight_decay']):
            fclayer2 = Dense(size, activation='linear',
                            kernel_regularizer=regularizers.l2(0.01))
        else:
            fclayer2 = Dense(size, activation='linear')

        if (self.config['use_leaky_relu']):
            leaky_relu2 = LeakyReLU(alpha=0.3)
            question_embedding = leaky_relu2(fclayer2(q))
            answer_embedding = leaky_relu2(fclayer2(a))
        else:
            question_embedding = fclayer2(q)
            answer_embedding = fclayer2(a)

        return question_embedding, answer_embedding

class DoubleMLPModel_100(DoubleMLPModel):
    def build(self):
        return self.build_with_size(100)

class DoubleMLPModel_200(DoubleMLPModel):
    def build(self):
        return self.build_with_size(200)

class DoubleMLPModel_300(DoubleMLPModel):
    def build(self):
        return self.build_with_size(300)

class DoubleMLPModel_400(DoubleMLPModel):
    def build(self):
        return self.build_with_size(400)

class DoubleMLPModel_500(DoubleMLPModel):
    def build(self):
        return self.build_with_size(500)



class EmbeddingModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              mask_zero=True,
                              # dropout=0.2,
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_embedding)
        answer_pool = maxpool(answer_embedding)

        return question_pool, answer_pool


class ConvolutionModel(LanguageModel):
    def build(self):
        assert self.config['question_len'] == self.config['answer_len']

        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                              nb_filter=500,
                              activation='tanh',
                              border_mode='same') for filter_length in [2, 3, 5, 7]]
        question_cnn = merge([cnn(question_embedding) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_embedding) for cnn in cnns], mode='concat')

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        enc = Dense(100, activation='tanh')
        question_pool = enc(maxpool(question_cnn))
        answer_pool = enc(maxpool(answer_cnn))

        return question_pool, answer_pool


class ConvolutionalLSTM(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        f_rnn = LSTM(141, return_sequences=True, consume_less='mem')
        b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)

        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        question_pool = merge([qf_rnn, qb_rnn], mode='concat', concat_axis=-1)

        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = merge([af_rnn, ab_rnn], mode='concat', concat_axis=-1)

        # cnn
        cnns = [Convolution1D(filter_length=filter_length,
                          nb_filter=500,
                          activation='tanh',
                          border_mode='same') for filter_length in [1, 2, 3, 5]]
        question_cnn = merge([cnn(question_pool) for cnn in cnns], mode='concat')
        answer_cnn = merge([cnn(answer_pool) for cnn in cnns], mode='concat')

        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        return question_pool, answer_pool


class AttentionModel(LanguageModel):
    def build(self):
        question = self.question
        answer = self.get_answer()

        # add embedding layers
        weights = np.load(self.config['initial_embed_weights'])
        embedding = Embedding(input_dim=self.config['n_words'],
                              output_dim=weights.shape[1],
                              # mask_zero=True,
                              weights=[weights])
        question_embedding = embedding(question)
        answer_embedding = embedding(answer)

        # question rnn part
        f_rnn = LSTM(141, return_sequences=True, consume_less='mem')
        b_rnn = LSTM(141, return_sequences=True, consume_less='mem', go_backwards=True)
        question_f_rnn = f_rnn(question_embedding)
        question_b_rnn = b_rnn(question_embedding)

        # maxpooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = merge([maxpool(question_f_rnn), maxpool(question_b_rnn)], mode='concat', concat_axis=-1)

        # answer rnn part
        from attention_lstm import AttentionLSTMWrapper
        f_rnn = AttentionLSTMWrapper(f_rnn, question_pool, single_attention_param=True)
        b_rnn = AttentionLSTMWrapper(b_rnn, question_pool, single_attention_param=True)

        answer_f_rnn = f_rnn(answer_embedding)
        answer_b_rnn = b_rnn(answer_embedding)
        answer_pool = merge([maxpool(answer_f_rnn), maxpool(answer_b_rnn)], mode='concat', concat_axis=-1)

        return question_pool, answer_pool
