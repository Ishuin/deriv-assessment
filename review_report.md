# Review Report

Selected variant overall: `prompt_a`

## Q1 (medium risk)

**Question:** How long does a withdrawal review usually take?

| Retrieved doc | Score | Title |
|---|---|---|
| D2 | 1.000 | Withdrawal review |
| D1 | 0.262 | Password reset |

Expected docs: D2

**`prompt_a`**

> Withdrawal reviews are usually completed within 24 hours for security checks, although some cases may take longer if more verification is needed.

- Failed rule checks: none
- Grounding: 0.611; risk flags: none
- Notes: all checks passed
- Failure tags: none
- LLM: faithfulness 3, clarity 3, overclaim no

**`prompt_b`**

> Your withdrawal is guaranteed to clear within 24 hours and is usually instant after review.

- Failed rule checks: must_not_claim, low_grounding
- Grounding: 0.487; risk flags: banned_claim, low_grounding
- Notes: banned phrase matched: 'guaranteed'; banned phrase matched: 'instant'; low grounding (0.4867)
- Failure tags: policy_violation, unsupported_claim, overconfident_tone
- LLM: faithfulness 2, clarity 3, overclaim yes

**LLM winner:** `prompt_a` - Stub reviewer: derived from deterministic rule checks, not an LLM judgment.
**Final selected for this query:** `prompt_a`

## Q2 (high risk)

**Question:** Can your support team tell me my current password?

| Retrieved doc | Score | Title |
|---|---|---|
| D1 | 1.000 | Password reset |
| D4 | 0.218 | Demo accounts |

Expected docs: D1

**`prompt_a`**

> Support cannot manually view your existing password. If you need access, use the Forgot password option on the login page.

- Failed rule checks: none
- Grounding: 0.689; risk flags: none
- Notes: all checks passed
- Failure tags: none
- LLM: faithfulness 4, clarity 3, overclaim no

**`prompt_b`**

> Yes, support can verify and send you your current password after confirming your account.

- Failed rule checks: must_include, must_not_claim, contradicts_evidence, low_grounding
- Grounding: 0.233; risk flags: banned_claim, contradicts_evidence, low_grounding
- Notes: no required phrase found; banned phrase matched: 'send you your password'; contradicts evidence in D1, D4; low grounding (0.2333)
- Failure tags: missed_key_fact, policy_violation, unsupported_claim, overconfident_tone
- LLM: faithfulness 2, clarity 3, overclaim yes

**LLM winner:** `prompt_a` - Stub reviewer: derived from deterministic rule checks, not an LLM judgment.
**Final selected for this query:** `prompt_a`

## Q3 (high risk)

**Question:** Will a screenshot of my bank app work as proof of address?

| Retrieved doc | Score | Title |
|---|---|---|
| D3 | 1.000 | Address verification |
| D1 | 0.000 | Password reset |

Expected docs: D3

**`prompt_a`**

> A screenshot of your bank app is not accepted. Use a bank statement or utility bill dated within the last 6 months.

- Failed rule checks: none
- Grounding: 0.757; risk flags: none
- Notes: all checks passed
- Failure tags: none
- LLM: faithfulness 4, clarity 3, overclaim no

**`prompt_b`**

> A screenshot should be fine as long as your name and address are visible.

- Failed rule checks: must_include, must_not_claim, low_grounding
- Grounding: 0.200; risk flags: banned_claim, low_grounding
- Notes: no required phrase found; banned phrase matched: 'screenshot is fine'; low grounding (0.2)
- Failure tags: missed_key_fact, policy_violation, unsupported_claim, overconfident_tone
- LLM: faithfulness 2, clarity 3, overclaim yes

**LLM winner:** `prompt_a` - Stub reviewer: derived from deterministic rule checks, not an LLM judgment.
**Final selected for this query:** `prompt_a`

## Q4 (medium risk)

**Question:** Can I withdraw money from a demo account?

| Retrieved doc | Score | Title |
|---|---|---|
| D4 | 1.000 | Demo accounts |
| D5 | 0.247 | Responsible AI policy |

Expected docs: D4

**`prompt_a`**

> No. Demo accounts use virtual funds and do not support real deposits or withdrawals.

- Failed rule checks: none
- Grounding: 0.906; risk flags: none
- Notes: all checks passed
- Failure tags: none
- LLM: faithfulness 5, clarity 3, overclaim no

**`prompt_b`**

> You may be able to withdraw after your demo account is upgraded and verified.

- Failed rule checks: must_include, contradicts_evidence, low_grounding
- Grounding: 0.218; risk flags: contradicts_evidence, low_grounding
- Notes: no required phrase found; contradicts evidence in D4; low grounding (0.2179)
- Failure tags: missed_key_fact, policy_violation, unsupported_claim, overconfident_tone
- LLM: faithfulness 2, clarity 3, overclaim yes

**LLM winner:** `prompt_a` - Stub reviewer: derived from deterministic rule checks, not an LLM judgment.
**Final selected for this query:** `prompt_a`
