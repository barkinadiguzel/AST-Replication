# AST-Replication
AST-Replication implements an asymmetric student-teacher framework where a normalizing flow teacher and a CNN student are trained to match outputs on normal data. At inference, anomalies are identified by measuring the deviation between student and teacher outputs, which increases for out-of-distribution samples.
