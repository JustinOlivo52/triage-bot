# Emergency Severity Index — Condensed Triage Reference

> **Status: DRAFT — pending clinical review.**
> This document was drafted as a condensed, self-authored summary of publicly
> described ESI v4 concepts. It is **not** a reproduction of the ESI
> Implementation Handbook or any other copyrighted material, and it is not a
> substitute for it. It exists to give the retrieval layer something accurate
> and redistributable to ground against.
>
> **Sources this summarises:** the Emergency Severity Index (ESI) v4 algorithm
> as published by the Agency for Healthcare Research and Quality (AHRQ) and
> maintained by the Emergency Nurses Association (ENA). Consult the current
> official handbook for authoritative criteria.
>
> All thresholds in this document must match `config.py`. If they diverge, the
> code is the source of truth and this document is wrong.

---

## Purpose of Triage

Triage assigns each arriving patient an acuity level so that limited clinical
resources reach the sickest patients first. It is a sorting decision, not a
diagnosis. The question is not "what is wrong with this patient" but "how long
can this patient safely wait, and what will they need".

Two failure modes are not symmetrical. **Under-triage** — assigning a lower
acuity than the patient's condition warrants — delays care for someone who is
deteriorating and is the failure that causes harm. **Over-triage** consumes
resources and lengthens waits for others. Where the evidence is genuinely
ambiguous, the safer error is to over-triage.

---

## The ESI Algorithm

ESI sorts patients into five levels using a sequence of four decision points.
The sequence matters: each question is only asked if the previous one was
answered "no".

### Decision Point A — Is this patient dying?

Does the patient require an immediate life-saving intervention? If yes, the
patient is **ESI 1** and the algorithm stops.

Life-saving interventions include airway management, assisted ventilation,
emergency medications for an unstable rhythm or pressure, and immediate
haemodynamic resuscitation. Ordinary diagnostic work-up is not a life-saving
intervention.

### Decision Point B — Should this patient wait?

Is this a high-risk situation, or is the patient confused, lethargic or
disoriented, or in severe pain or distress? If yes, the patient is **ESI 2**.

A high-risk situation is one where a plausible worst-case outcome is serious
and time-dependent, even if the patient currently looks well. The judgment is
about trajectory and risk, not present appearance.

### Decision Point C — How many resources will this patient need?

For patients who are not ESI 1 or 2, acuity is predicted by the number of
distinct resources the work-up will consume.

- Two or more resources → **ESI 3**
- Exactly one resource → **ESI 4**
- No resources → **ESI 5**

### Decision Point D — Do the vital signs change the answer?

For a patient otherwise scored ESI 3, out-of-range vital signs should prompt
consideration of upgrading to ESI 2. Danger-zone vitals are a reason to
re-examine the assignment, not an automatic reassignment.

---

## ESI Levels

### ESI 1 — Resuscitation

Requires immediate life-saving intervention. Physician at the bedside
immediately. No waiting room time under any circumstances.

Typical presentations: cardiac or respiratory arrest, unresponsiveness,
severe respiratory distress with inadequate oxygenation, profound
haemodynamic instability, unstable airway.

### ESI 2 — Emergent

High risk, or altered mental status, or severe pain or distress. Should not
wait. Placed in a treatment area as soon as one is available, ahead of all
lower-acuity patients.

Typical presentations: chest pain with features suggesting an acute coronary
syndrome, stroke symptoms within the treatment window, suspected sepsis,
significant mechanism of injury, new confusion or lethargy, immunocompromised
patient with fever, suspected ectopic pregnancy, active suicidal ideation.

### ESI 3 — Urgent

Stable, but the work-up is expected to require two or more resources. This is
the most common level in most departments and also the level where acuity is
most often underestimated, particularly when vital signs are abnormal.

Typical presentations: abdominal pain requiring laboratory work and imaging,
moderate injury requiring imaging and procedural care, vomiting requiring
intravenous fluids and laboratory work.

### ESI 4 — Less Urgent

Stable, expected to require exactly one resource.

Typical presentations: simple laceration requiring repair, isolated limb
injury requiring a single radiograph, uncomplicated urinary symptoms requiring
a single test.

### ESI 5 — Non-Urgent

Stable, expected to require no resources beyond examination.

Typical presentations: prescription refill, suture or dressing removal, minor
superficial complaint, a recheck with no new findings.

---

## What Counts as a Resource

A resource is a distinct diagnostic or therapeutic activity that consumes
departmental capacity beyond examination and simple advice.

**Counts as a resource:** laboratory studies, electrocardiogram, diagnostic
imaging, intravenous fluids, intravenous, intramuscular or nebulised
medication, specialty consultation, a simple or complex procedure.

**Does not count:** history and physical examination, point-of-care testing
performed at the bedside, oral medication, a prescription, a tetanus
immunisation, a dressing change, crutches or a sling.

Counting is by **type**, not by quantity. Three separate blood tests are one
resource. A blood test plus an imaging study is two.

---

## Vital Sign Danger Zones

These thresholds are adult values and are duplicated in `config.py`, which is
the operative source. They are applied in two tiers.

**Concerning** — outside normal range; a reason to look more closely, not by
itself a statement that the patient cannot wait:
heart rate above 100 or below 60; respiratory rate above 20 or below 12;
oxygen saturation below 94%; temperature above 38.5 °C or below 36.0 °C;
systolic pressure above 180 or below 100; diastolic pressure above 110.

**Critical** — derangement warranting immediate physician attention regardless
of the assigned acuity level:
heart rate above 130 or below 45; respiratory rate above 30 or below 8;
oxygen saturation below 90%; temperature above 40.0 °C or below 35.0 °C;
systolic pressure above 220 or below 90; diastolic pressure above 120.

**These ranges do not apply to children.** Normal paediatric heart and
respiratory rates are substantially higher and vary by age; a well three-year-old
sits near a heart rate of 110 and a respiratory rate of 26, both of which read
as abnormal against adult values. Age-banded thresholds are a known gap in the
current implementation.

---

## High-Risk Presentations

These warrant consideration of ESI 2 even when the patient appears well and
the vital signs are unremarkable.

**Cardiac:** chest pain or pressure with exertional features, radiation,
diaphoresis, or associated dyspnoea. Anginal equivalents in diabetic, elderly
and female patients often present without classic chest pain.

**Neurological:** any focal deficit of sudden onset — facial droop, unilateral
weakness, speech disturbance — is time-critical. New confusion in an older
patient is a presenting sign of serious illness, not a baseline finding to be
assumed.

**Infectious:** fever with tachycardia and tachypnoea suggests sepsis.
Immunocompromised patients and those at the extremes of age may mount no
fever at all.

**Respiratory:** inability to speak in full sentences, accessory muscle use,
or oxygen saturation below the concerning threshold.

**Obstetric:** abdominal pain or bleeding in a patient of childbearing age
requires exclusion of ectopic pregnancy.

**Psychiatric:** active suicidal or homicidal ideation requires immediate
safety assessment regardless of physical presentation.

---

## Age as a Risk Modifier

Age alone does not determine acuity. Age combined with any other abnormal
finding raises it, because both extremes of the age range blunt the usual
warning signs.

**Paediatric patients (roughly 5 and under)** cannot reliably report symptoms,
compensate well until they decompensate abruptly, and have age-specific normal
ranges that adult thresholds misread.

**Geriatric patients (roughly 65 and over)** frequently present atypically.
Myocardial infarction without chest pain, infection without fever, and acute
abdomen with minimal tenderness are all common. Baseline medication can blunt
the tachycardic response that would otherwise signal instability.

---

## Documenting the Decision

A triage assignment should record what was observed, which criterion applied,
and what is expected to be needed. An assignment that cannot be explained
cannot be reviewed, and an assignment that cannot be reviewed cannot be
improved.

Negative findings matter as much as positive ones. "Denies chest pain" is a
clinically meaningful statement and is not the same as chest pain. A history
of a condition is not the same as that condition being present now.
