#!/bin/bash

python experiments.py -ds darkreddit  -ntr 1000 -nte 250 -nrep 15
python experiments.py -ds silkroad  -ntr 1000 -nte 250 -nrep 15
python experiments.py -ds agora  -ntr 1000 -nte 250 -nrep 15
python experiments.py -ds amazon  -ntr 1000 -nte 250 -nrep 15