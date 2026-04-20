from src.models.attention import BahdanauAttention, LuongAttention
from src.models.decoder import Decoder
from src.models.encoder import Encoder
from src.models.layers import Embedding, GRU
from src.models.seq2seq import Seq2Seq


def build_model(src_vocab, tgt_vocab, embedding_dim, hidden_size, device, use_attention=True, attention_type='luong'):
    src_embedding = Embedding(len(src_vocab), embedding_dim)
    tgt_embedding = Embedding(len(tgt_vocab), embedding_dim)
    encoder_gru = GRU(input_size=embedding_dim, hidden_size=hidden_size)
    decoder_gru = GRU(input_size=embedding_dim, hidden_size=hidden_size)
    
    if use_attention:
        if attention_type.lower() == 'bahdanau':
            attention = BahdanauAttention(hidden_size=hidden_size)
        elif attention_type.lower() == 'luong':
            attention = LuongAttention(hidden_size=hidden_size)
        else:
            raise ValueError(f"Unsupported attention_type: {attention_type}")
    else:
        attention = None

    encoder = Encoder(rnn_model=encoder_gru)
    decoder = Decoder(rnn_model=decoder_gru, attention_model=attention, output_vocab_size=len(tgt_vocab))
    model = Seq2Seq(
        src_embedding=src_embedding,
        tgt_embedding=tgt_embedding,
        encoder=encoder,
        decoder=decoder,
        sos_token_id=tgt_vocab.SOS_IDX,
        eos_token_id=tgt_vocab.EOS_IDX,
    )
    return model.to(device)
