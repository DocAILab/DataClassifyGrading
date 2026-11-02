"""Three auditable episode rewards; labels are consumed only after rollout."""


def outcome(trace, reference, native_guard_valid):
    return int(bool(trace.get('terminated') and native_guard_valid
                    and trace.get('final_result') is not None
                    and trace['final_result'] == reference))


def combine(correct, invalid_count, dependency_count, budget=32):
    if correct not in (0, 1) or type(budget) is not int or budget <= 0:
        raise ValueError('binary outcome and fixed positive budget required')
    if any(type(x) is not int or not 0 <= x <= budget for x in (invalid_count, dependency_count)):
        raise ValueError('reward counts outside actual Skill budget')
    i, d = invalid_count / budget, dependency_count / budget
    return dict(outcome=correct,invalid=i,dependency=d,
                A1=float(correct),A2=correct-.3*i,A3=correct-.3*i+.2*d,
                A4=correct-.3*i+.2*correct*d)


class QueryLedger:
    def __init__(self): self.known = {}

    def before(self, key, source_version):
        return self.known.get((key, source_version)) in ('obtained', 'missing')

    def after(self, key, source_version, status):
        if status in ('obtained', 'missing'):
            self.known[key, source_version] = status

