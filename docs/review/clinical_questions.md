# Questions for the clinician

Design questions that need a clinical answer before release. Please write your answer and
any reasoning under each question.

## 1. High-alert medicines with a missing dose

In rules mode (no model), scribecheck reports a medicine whose dose the transcript gives but
the note leaves out as a warning of **supporting** severity, the same as a missing medicine
or frequency. A changed dose is **critical**. In full mode, the model's own severity grade
replaces this default once it confirms the omission.

Should a missing dose of a high-alert medicine (for example insulin, anticoagulants or
opioids) be **critical** even in rules mode?

If yes, please say which medicine classes the list should cover, or point to a list you
trust (for example ISMP's high-alert medications). No high-alert list is built yet.

**Answer:**
