import unittest

import jax
from jax import numpy as jnp
from jax import grad

import numpy.random as npr

from sepia.seq.transformer import Transformer

from sepia.seq.dataloader import SeqDataLoader

class TestTransformer(unittest.TestCase):

    def setUp(self):
        pass

    def test_fit(self):
        model = Transformer(decoder_size=1, encoder_size=1)

        ha_tag = "YPYDVPDYA".lower()

        dataset = [ha_tag]

        token_dict = model.get_token_dict()
        seq_length = model.get_seq_length()
        token_dim = model.get_token_dim()

        dataloader = SeqDataLoader(token_dict, seq_length, token_dim, dataset=dataset, batch_size=1)

        model.fit(dataloader, max_epochs=2)

    def test_multihead(self):

        my_seq = "arndcqeghilkmfpstwyvuox"
        seq_length = len(my_seq)
        batch_size = 4
        encoder_size = 2
        decoder_size = 2
        for number_heads in [1,3,4]:
            model = Transformer(number_heads=number_heads, \
                    seq_length=seq_length,\
                    encoder_size=encoder_size, \
                    decoder_size=decoder_size)

            my_input = [my_seq] * batch_size

            output_single = model(my_input[0:1])
            output_batch = model(my_input)
            output_string = model(my_input[0])

            self.assertEqual(output_single[0], output_batch[0])
            self.assertEqual(output_single[0], output_string[0])
            self.assertEqual(output_single, output_batch[0:1])

    def test_batch_size(self):

        my_seq = "arndcqeghilkmfpstwyvuox"
        seq_length = len(my_seq)
        for encoder_size in [1,3,4]:
            for decoder_size in [1,3,4]:
                model = Transformer(seq_length=seq_length,\
                        encoder_size=encoder_size, decoder_size=decoder_size)
                for batch_size in [1,3,4]:

                    my_input = [my_seq] * batch_size

                    output_single = model(my_input[0:1])
                    output_batch = model(my_input)
                    output_string = model(my_input[0])

                    self.assertEqual(output_single[0], output_batch[0])
                    self.assertEqual(output_single[0], output_string[0])
                    self.assertEqual(output_single, output_batch[0:1])


if __name__ == "__main__":
    
    unittest.main(verbosity=2)
