"""R12 science and authorized200 source identity, no new prompt tuning."""
from r14_common import *
import native200
import verify200


def runtime():
    return native200.runtime()


def bundle():
    return native200.bundle()


def build(engine, scoped, stopped):
    return native200.build(engine, scoped, stopped, bundle())


def verify(target, receipt):
    proof = verify200.verify(target, receipt)
    if proof['summary'].get('TestOnly') is True:
        raise ValueError('QA evidence cannot become a formal paid source proof')
    return proof
