import tensorflow as tf
from tensorflow.keras.layers import LSTM, Embedding, Dense, Input
from tensorflow.keras import Model

from pudb import set_trace
from tensorflow.keras.utils import plot_model


class EncoderDecoderLSTM:
    def __init__(
        self,
        encoder_vocabulary_size,
        encoder_lstm_units,
        encoder_latent_dimension,
        decoder_vocabulary_size,
        decoder_lstm_units,
        decoder_latent_dimension,
    ):
        encoder_input = Input(shape=(None,), name="Encoder_input")
        encoder_embedding = Embedding(
            input_dim=encoder_vocabulary_size,
            output_dim=encoder_latent_dimension,
            name="Encoder_embedding",
        )(encoder_input)
        encoder_LSTM = LSTM(
            units=encoder_lstm_units, return_state=True, name="Encoder_LSTM"
        )
        encoder_output, encoder_state_h, encoder_state_c = encoder_LSTM(
            encoder_embedding
        )
        self.inference_encoder_model = Model(
            inputs=encoder_input, outputs=[encoder_state_h, encoder_state_c]
        )

        decoder_input = Input(shape=(None,), name="Decoder_input")
        decoder_embedding = Embedding(
            input_dim=decoder_vocabulary_size,
            output_dim=decoder_latent_dimension,
            name="Decoder_embedding",
        )(decoder_input)
        decoder_LSTM = LSTM(
            units=decoder_lstm_units,
            return_sequences=True,
            return_state=True,
            name="Decoder_LSTM",
        )
        decoder_output, decoder_state_h, decoder_state_h = decoder_LSTM(
            decoder_embedding, initial_state=[encoder_state_h, encoder_state_c]
        )
        decoder_output = Dense(
            decoder_vocabulary_size, activation="softmax", name="Decoder_dense_output"
        )(decoder_output)
        self.train_model = Model(
            inputs=[encoder_input, decoder_input], outputs=decoder_output
        )
        self.train_model.compile(
            optimizer="rmsprop", loss="categorical_crossentropy", metrics=["acc"]
        )

        decoder_state_h_input = Input(
            shape=(decoder_lstm_units,), name="Decoder_state_h_input"
        )
        decoder_state_c_input = Input(
            shape=(decoder_lstm_units,), name="Decoder_state_c_input"
        )
        decoder_inference_output, decoder_inference_state_h, decoder_inference_state_c = decoder_LSTM(
            decoder_embedding,
            initial_state=[decoder_state_h_input, decoder_state_c_input],
        )
        self.inference_decoder_model = Model(
            inputs=[decoder_input, decoder_state_h_input, decoder_state_c_input],
            outputs=[
                decoder_inference_output,
                decoder_inference_state_h,
                decoder_inference_state_c,
            ],
        )

    def plot_models(self):
        plot_model(self.train_model, show_shapes=True, to_file="plots/train_model.png")
        plot_model(
            self.inference_encoder_model,
            show_shapes=True,
            to_file="plots/inference_encoder_model.png",
        )
        plot_model(
            self.inference_decoder_model,
            show_shapes=True,
            to_file="plots/inference_decoder_model.png",
        )


if __name__ == "__main__":
    estimator = EncoderDecoderLSTM(
        encoder_vocabulary_size=20,
        encoder_lstm_units=10,
        encoder_latent_dimension=32,
        decoder_vocabulary_size=20,
        decoder_lstm_units=10,
        decoder_latent_dimension=32,
    )
    estimator.plot_models()
    set_trace()
