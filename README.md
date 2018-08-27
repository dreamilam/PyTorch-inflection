# Universal Morphological reinfection : CU Boulder's submission to CONLL Sigmorphon 2018 <h1> 

We present two neural systems for the Subtask 2 of CONLL shared task on Universal Morphological reinflection. Both are implementations of encoder decoder RNNs. The first system is a standard encoder decoder network with a soft attention mechanism.
The second system first tries to predict the morpho-syntactic descriptions(MSDs) for the target word and then inflects the lemma based on the predicted MSD. This uses two separate encoder-decoder networks for the MSD prediction and the subsequent inflection.

The inflection model is based on CU Boulder's submission to Sigmorphon 2017 and is directly adapted from https://github.com/Adamits/conll2018/tree/master/task1/deep

To run the code, use the following command

python3 baseline/baseline.py trainsets/en-track1-high devsets/en-track1-covered devsets/en-uncovered
