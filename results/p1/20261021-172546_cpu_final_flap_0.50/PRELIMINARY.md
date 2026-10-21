# PRELIMINARY — not used in the report

Measured on AC power ("Best performance"), but processor performance varied between 31% and 61% during the
session, and the measurement was disturbed by other work on the laptop (a heavy library import and test runs
overlapped the FLAP-35% and FLAP-20% runs; two runs were made with uncommitted code). The variants are
non-monotonic in size (FLAP-20% decode 470 ms/token vs 443 ms unpruned). CPU timing is deferred to the later
speed-up step (int8 weights, P-core thread pinning, dedicated idle measurement window).
