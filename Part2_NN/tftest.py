import keras
from keras.models import Model
from keras.layers import Input, LSTM, Dense

batch_size = 64  # batch size for training
epochs = 100  # number of epochs to train for
latent_dim = 250  # latent dimensionality of the encoding space

encoder_inputs = Input(shape=(None, 50))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None, 100))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(100, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)

model.compile(optimizer='adam', loss='categorical_crossentropy')
model.summary()


print("OK")
