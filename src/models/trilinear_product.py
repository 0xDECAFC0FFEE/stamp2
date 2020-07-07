import tensorflow as tf

@tf.function()
def trilinear_product(a, b, c):
    """gets the trilinear product of a, b, and c.
    if a, b, c are [1, d] vectors, <a, b, c> = \Sigma_i a_i * b_i * c_i

    note that the ordering of a, b, and c in the paper is slightly different than
    the implementation here as 1) it disagrees with the actual source code released and
    2) it results in shape errors

    Args:
        a (matrix): [m, d] matrix of average of the first n-1 session item embeddings
        b (matrix): [m, d] matrix of last session item embeddings
        c (matrix): [n, d] matrix of all the item embeddings

    Returns:
        matrix: [m, n] matrix of item scores for each session
    """
    return tf.transpose(c @ tf.transpose(a * b))