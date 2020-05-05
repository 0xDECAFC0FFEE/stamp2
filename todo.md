# todo
1. fix VectorizableNotDumbSparseTopKCategoricalAccuracy
    1. add small noise to output logits and use SparseTopKCategoricalAccuracy
2. implement nce loss. replace the softmax layer
3. clustering (~64 clusters) ( equation 2 in https://arxiv.org/pdf/1611.01144.pdf). each g_i is a random number generated for each training batch.
    1. reason for clustering is to leverage session information in other training example sessions in the same cluster to output the session information
4. replace binning with masking and variable session lengths