# Place the dataset here

This directory is intentionally empty in version control. The MQTT-IoT-IDS2020
licence does not grant redistribution, so the data is not vendored.

1. Create a free IEEE account and download the dataset from
   https://ieee-dataport.org/open-access/mqtt-iot-ids2020-mqtt-internet-things-intrusion-detection-dataset
   (DOI 10.21227/bhxy-ep04).
2. Copy `uniflow_features.zip` (16.08 MB) and `biflow_features.zip` (14.56 MB)
   into this directory. The loader unpacks archives automatically, or you may
   extract them yourself.
3. Run `python -m experiments.run_all --phases audit`.

The audit records a SHA256 for every archive and every CSV, so a future reader
can confirm they used the same files.
