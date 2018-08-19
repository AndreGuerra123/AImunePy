from keras.layers import Input, BatchNormalization, Bidirectional, Concatenate, Bidirectional, Dense, Dropout, LSTM
from keras.models import Model


def soft_attention_alignment(input_1, input_2):

    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
    output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2,1))(Lambda(lambda x: softmax(x, axis=2),
    output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned

def esim(maxlen=32,
    lstm_dim=300,
    dense_dim=300,
    dense_dropout=0.5):

    q1 = Input(name='q1',shape=(maxlen,))
    q2 = Input(name='q2',shape=(maxlen,))

    bn = BatchNormalization(axis=2)
    #q1_embed = bn(embedding_layer(q1)) PLEASE PROVIDE THE FULL CODE
    q1_embed = bn(q1)
    #q2_embed = bn(embedding_layer(q2)) PLEASE PROVIDE THE FULL CODE
    q2_embed = bn(q2)

    encode = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)


    q1_combined = Concatenate()([q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()([q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(LSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)


    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])


    merged = Concatenate()([q1_rep, q2_rep])

    dense = BatchNormalization()(merged)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='elu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(heuristic_score, activation='softmax')(dense)
    return Model(inputs=[q1, q2], outputs=out_)


model = esim()    
model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['categorical_crossentropy','accuracy'])
