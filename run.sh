#!/bin/bash
# probs=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.2 1.3 1.4 1.5)
probs=(1.1 1.2)

for i in "${probs[@]}"; do
    echo $i >> guru99.txt
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    python3 working_code.py True $i
    echo "" >> guru99.txt
done
