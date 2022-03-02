import os
import sys
from tensorflow.python.summary.summary_iterator import summary_iterator

LOG_DIR = "./log_dir/piunet"
CHECKPOINT = sys.argv[1]
OUT_FILE = sys.argv[2]

# List of cPSNR & cSSIM computed on validation set, in order of training.
cPSNR = []
cSSIM = []

# Get TFEvent name
for summary in os.listdir(f"{LOG_DIR}/{CHECKPOINT}"):
    path = f"{LOG_DIR}/{CHECKPOINT}/{summary}"
    for event in summary_iterator(path):
        print(event)
        for v in event.summary.value:
            if v.tag == "val/cPSNR":
                cPSNR.append(v.simple_value)
            elif v.tag == "val/cSSIM":
                cSSIM.append(v.simple_value)
            

print("cPSNR on validation set (last round of validation):", cPSNR[-1])
print("cSSIM on validation set (last round of validation):", cSSIM[-1])
# Save results
with open(OUT_FILE, 'w') as f:
    f.write(f"cPSNR: {cPSNR[-1]}\n")
    f.write(f"cSSIM: {cSSIM[-1]}\n")
print(f"Results saved to {OUT_FILE}")