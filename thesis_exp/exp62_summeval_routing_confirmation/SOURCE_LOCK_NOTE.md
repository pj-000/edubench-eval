# Exp62 transitive source-lock note

The current repository root contains the Exp63 runtime version of
`thesis_exp/exp57_cbrd/method.py`, whose SHA-256 is
`00f044bc764668910a6ff6c809d481acfd8e75a29974df3d0b71ace9728f68c8`.

Exp62 was executed before that file's trailing blank line was normalized.  Its
formal source lock therefore correctly records SHA-256
`d112992222bc5166068fbc887e4dbdadb16f437faf24cecf5d50693efb0a5fa7`.
The exact Exp62 runtime bytes are preserved at
`runtime_snapshot/exp57_method.py` and match that hash.  A diff between the
snapshot and the current root file contains only one additional blank line at
end of file; executable Python tokens and behavior are identical.

This snapshot is supplied to make both experiments byte-auditable rather than
rewriting either historical source lock after results were observed.

