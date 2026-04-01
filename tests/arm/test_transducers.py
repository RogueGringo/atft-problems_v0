import numpy as np
import pytest

def test_text_transducer_simple():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("Hi.")
    assert pc.data.shape == (3, 4)
    assert pc.source == "text:3chars"

def test_text_transducer_channels():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("A b")
    assert pc.data[0, 1] == 3   # 'A' → upper=3
    assert pc.data[1, 1] == 0   # ' ' → space=0
    assert pc.data[2, 1] == 1   # 'b' → lower=1

def test_text_transducer_punctuation():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("a,b.")
    assert pc.data[1, 3] == 1  # ',' → comma=1
    assert pc.data[3, 3] == 3  # '.' → stop=3

def test_text_transducer_paragraph():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc = t.transduce("a\n\nb")
    assert pc.data[1, 2] == 3  # first \n → paragraph=3
    assert pc.data[2, 2] == 3  # second \n → paragraph=3

def test_text_transducer_hash_deterministic():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    pc1 = t.transduce("Hello World")
    pc2 = t.transduce("Hello World")
    assert pc1.hash == pc2.hash

def test_text_transducer_describe():
    from arm.void.transducers import TextTransducer
    t = TextTransducer()
    desc = t.describe()
    assert "text" in desc.lower()
    assert "4" in desc

def test_generic_transducer_csv_string():
    from arm.void.transducers import GenericTransducer
    t = GenericTransducer()
    csv_data = "1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0"
    pc = t.transduce(csv_data)
    assert pc.data.shape == (3, 3)
    assert np.isclose(pc.data[0, 0], 1.0)

def test_veilbreak_transducer_from_cached_data():
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    experiments = [
        {"laser_wavelength": 650, "substance_dose": 30, "laser_class": 3,
         "substance": "N,N-DMT", "observed": True,
         "description": "Saw patterns in laser field"},
        {"laser_wavelength": 532, "substance_dose": 0, "laser_class": 2,
         "substance": "none", "observed": False,
         "description": "Control run no substance"},
    ]
    pc = t.transduce_experiments(experiments)
    assert pc.data.shape[0] == 2
    assert pc.data.shape[1] >= 4

def test_veilbreak_transducer_describe():
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    desc = t.describe()
    assert "veilbreak" in desc.lower()

def test_veilbreak_text_channel():
    from arm.void.transducers import VeilbreakTransducer
    t = VeilbreakTransducer()
    experiments = [
        {"laser_wavelength": 650, "substance_dose": 30, "laser_class": 3,
         "substance": "N,N-DMT", "observed": True,
         "description": "Geometric patterns emerged"},
    ]
    pc_struct, pc_text = t.transduce_multichannel(experiments)
    assert pc_struct.data.shape[0] == 1
    assert pc_text.data.shape[0] > 0
