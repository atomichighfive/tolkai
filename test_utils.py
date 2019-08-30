def drop_to_pudb():
    from pudb import set_trace
    from utils import load_data, TextCoder

    data = load_data(
        # ["data/europarl-v7.sv-en.sv", "data/europarl-v7.sv-en.en"],
        ["test_data/10k.sv", "test_data/10k.en"],
        # ["data/test.sv", "data/test.en"],
        ["sv", "en"],
    )
    sv_coder = TextCoder(data["sv"])
    en_coder = TextCoder(data["en"])
    set_trace()


def test_TextEncoder_encodeOneHot_outputShapeIsCorrect():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is a european session"], one_hot=True)
    assert encoded[0].shape == (7, 2000)


def test_TextEncoder_encodeOrdinal_outputShapeIsCorrect():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is a european session"], one_hot=False)
    assert encoded[0].shape == (7, 1)


def test_TextEncoder_encode_startAddedIfMissing():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is a european session"], one_hot=False)
    assert encoded[0][0, 0] == coder.word2int["_START_"]

    encoded = coder.encode(["this is a european session"], one_hot=True)
    assert encoded[0][0, :].argmax() == coder.word2int["_START_"]


def test_TextEncoder_encode_startNotAddedIfPresent():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["_START_ this is a european session"], one_hot=False)
    assert encoded[0][0, 0] == coder.word2int["_START_"]
    assert encoded[0][1, 0] != coder.word2int["_START_"]

    encoded = coder.encode(["_START_ this is a european session"], one_hot=True)
    assert encoded[0][0, :].argmax() == coder.word2int["_START_"]
    assert encoded[0][1, :].argmax() != coder.word2int["_START_"]


def test_TextEncoder_encode_stopAddedIfMissing():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is a european session"], one_hot=False)
    assert encoded[0][-1, 0] == coder.word2int["_STOP_"]

    encoded = coder.encode(["this is a european session"], one_hot=True)
    assert encoded[0][-1, :].argmax() == coder.word2int["_STOP_"]


def test_TextEncoder_encode_stopNotAddedIfPresent():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is a european session _STOP_"], one_hot=False)
    assert encoded[0][-1, 0] == coder.word2int["_STOP_"]
    assert encoded[0][-2, 0] != coder.word2int["_STOP_"]

    encoded = coder.encode(["this is a european session _STOP_"], one_hot=True)
    assert encoded[0][-1, :].argmax() == coder.word2int["_STOP_"]
    assert encoded[0][-2, :].argmax() != coder.word2int["_STOP_"]


def test_TextEncoder_encodeDecode_recoversOriginal():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    encoded = coder.encode(["this is it"], one_hot=False)
    decoded = coder.decode(encoded, one_hot=False)
    assert decoded[0] == "_START_ this is it _STOP_"


def test_TextEncoder_encodeDecode_oneHotAndOrdinalSame():
    from utils import load_data, TextCoder

    data = load_data(["test_data/10k.en"], ["en"], quiet=True)
    coder = TextCoder(data["en"], vocabulary_size=2000)
    onehot_encoded = coder.encode(data["en"].values, one_hot=True)
    onehot_decoded = coder.decode(onehot_encoded, one_hot=True)
    ordinal_encoded = coder.encode(data["en"].values, one_hot=False)
    ordinal_decoded = coder.decode(ordinal_encoded, one_hot=False)
    for onehot, ordinal in zip(onehot_decoded, ordinal_decoded):
        assert onehot == ordinal
