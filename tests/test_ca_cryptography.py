import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'packages', 'ca_cryptography', 'src'))

from ca import OneDimensionalCA
from rsrg import RandomStateRuleGen
from converter import Conv
from enc_dec import Encrypt, Decrypt

PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"

def check(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return condition


# ── Converter tests ──────────────────────────────────────────────────────────

def test_binary_string_length():
    c = Conv()
    binary = c.binary_string("Hi")
    check("'Hi' encodes to 16 bits (2 chars × 8)", len(binary) == 16, f"got {len(binary)}")

def test_binary_roundtrip():
    c = Conv()
    text = "Hello"
    binary = c.binary_string(text)
    recovered = c.convert_binary_to_text(binary)
    check("Text → binary → text roundtrip", recovered == text, f"got '{recovered}'")

def test_binary_roundtrip_special_chars():
    c = Conv()
    text = "CA!@#123"
    binary = c.binary_string(text)
    recovered = c.convert_binary_to_text(binary)
    check("Special chars roundtrip", recovered == text, f"got '{recovered}'")


# ── Encrypt / Decrypt tests ───────────────────────────────────────────────────

def test_encrypt_decrypt_roundtrip():
    e = Encrypt()
    d = Decrypt()
    key   = "10110010101101001011010"
    plain = "10101010101010101010101"
    enc = e.encrypt_text_with_final_state(key, plain)
    dec = d.decrypt_text_with_final_state(key, enc)
    check("Encrypt then decrypt recovers plaintext", dec == plain)

def test_xor_self_inverse():
    e = Encrypt()
    d = Decrypt()
    key  = "1100110011001100"
    msg  = "1010101010101010"
    enc  = e.encrypt_text_with_final_state(key, msg)
    dec  = d.decrypt_text_with_final_state(key, enc)
    check("XOR is self-inverse (enc→dec = identity)", dec == msg)

def test_encrypt_changes_message():
    e = Encrypt()
    key  = "11001100"
    msg  = "10101010"
    enc  = e.encrypt_text_with_final_state(key, msg)
    check("Encrypted output differs from plaintext", enc != msg)

def test_length_mismatch_raises():
    e = Encrypt()
    raised = False
    try:
        e.encrypt_text_with_final_state("101", "10101")
    except ValueError:
        raised = True
    check("Mismatched lengths raise ValueError", raised)


# ── CA evolution tests ────────────────────────────────────────────────────────

def test_ca_evolve_length():
    initial = "10110101001101"
    ca = OneDimensionalCA(rule_number=30, width=len(initial), generations=10)
    ca.initialize(initial)
    final = ca.evolve()
    check("Final state length equals initial state length",
          len(final) == len(initial), f"got {len(final)}")

def test_ca_deterministic():
    # Same rule/initial/generations must always produce the same final state
    initial = "10110101001101"
    def run():
        ca = OneDimensionalCA(rule_number=90, width=len(initial), generations=20)
        ca.initialize(initial)
        return ca.evolve()
    check("CA evolution is deterministic", run() == run())

def test_ca_different_rules_differ():
    initial = "10110101001101"
    def run(rule):
        ca = OneDimensionalCA(rule_number=rule, width=len(initial), generations=20)
        ca.initialize(initial)
        return ca.evolve()
    check("Different rules produce different outputs", run(30) != run(90))


# ── Full pipeline tests ────────────────────────────────────────────────────────

def _run_pipeline(plaintext):
    c = Conv()
    e = Encrypt()
    d = Decrypt()
    gen = RandomStateRuleGen()

    bin_str   = c.binary_string(plaintext)
    rule      = gen.RandomR()
    gens      = gen.RandomG()
    init      = gen.RandomS(len(bin_str))

    ca = OneDimensionalCA(rule, len(init), gens)
    ca.initialize(init)
    final = ca.evolve()

    enc_bin  = e.encrypt_text_with_final_state(final, bin_str)
    dec_bin  = d.decrypt_text_with_final_state(final, enc_bin)
    dec_text = c.convert_binary_to_text(dec_bin)
    return dec_text, plaintext

def test_pipeline_short():
    recovered, original = _run_pipeline("Hi")
    check("Full pipeline roundtrip — short string", recovered == original,
          f"expected '{original}' got '{recovered}'")

def test_pipeline_sentence():
    recovered, original = _run_pipeline("The quick brown fox")
    check("Full pipeline roundtrip — sentence", recovered == original,
          f"expected '{original}' got '{recovered}'")

def test_pipeline_numbers():
    recovered, original = _run_pipeline("12345 67890")
    check("Full pipeline roundtrip — numbers", recovered == original,
          f"expected '{original}' got '{recovered}'")


# ── Run all ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n=== Converter ===")
    test_binary_string_length()
    test_binary_roundtrip()
    test_binary_roundtrip_special_chars()

    print("\n=== Encrypt / Decrypt ===")
    test_encrypt_decrypt_roundtrip()
    test_xor_self_inverse()
    test_encrypt_changes_message()
    test_length_mismatch_raises()

    print("\n=== CA Evolution ===")
    test_ca_evolve_length()
    test_ca_deterministic()
    test_ca_different_rules_differ()

    print("\n=== Full Pipeline ===")
    test_pipeline_short()
    test_pipeline_sentence()
    test_pipeline_numbers()
    print()
