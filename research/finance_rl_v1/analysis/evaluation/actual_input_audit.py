import hashlib


def validate_actual_input(rewritten, actual, raw_bytes):
    # The model recorder sorts mapping keys when persisting JSON. Content and
    # message order stay exact; the recorded hash binds the pre-save order.
    if rewritten != actual['messages']:
        raise ValueError('actual request content differs')
    if hashlib.sha256(raw_bytes(rewritten)).hexdigest() != actual['messages_sha256']:
        raise ValueError('actual request hash differs')
