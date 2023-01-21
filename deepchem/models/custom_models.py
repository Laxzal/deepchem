from collections.abc import Sequence as SequenceCollection

import deepchem as dc
import numpy as np
import tensorflow as tf

from typing import List, Union, Tuple, Iterable, Dict, Optional
from deepchem.utils.typing import OneOrMany, LossFn, ActivationFn
from deepchem.data import Dataset, NumpyDataset, pad_features
from deepchem.feat.graph_features import ConvMolFeaturizer
from deepchem.feat.mol_graphs import ConvMol
from deepchem.metrics import to_one_hot
from deepchem.models import KerasModel, layers
from deepchem.models.losses import L2Loss, SoftmaxCrossEntropy, Loss
from deepchem.trans import undo_transforms
from tensorflow.keras.layers import Input, Dense, Reshape, Softmax, Dropout, Activation, BatchNormalization

class EnhancedWeaveModel(KerasModel):
  """Implements Google-style Weave Graph Convolutions

  This model implements the Weave style graph convolutions
  from [1]_.

  The biggest difference between WeaveModel style convolutions
  and GraphConvModel style convolutions is that Weave
  convolutions model bond features explicitly. This has the
  side effect that it needs to construct a NxN matrix
  explicitly to model bond interactions. This may cause
  scaling issues, but may possibly allow for better modeling
  of subtle bond effects.

  Note that [1]_ introduces a whole variety of different architectures for
  Weave models. The default settings in this class correspond to the W2N2
  variant from [1]_ which is the most commonly used variant..

  Examples
  --------

  Here's an example of how to fit a `WeaveModel` on a tiny sample dataset.

  >>> import numpy as np
  >>> import deepchem as dc
  >>> featurizer = dc.feat.WeaveFeaturizer()
  >>> X = featurizer(["C", "CC"])
  >>> y = np.array([1, 0])
  >>> dataset = dc.data.NumpyDataset(X, y)
  >>> model = dc.models.WeaveModel(n_tasks=1, n_weave=2, fully_connected_layer_sizes=[2000, 1000], mode="classification")
  >>> loss = model.fit(dataset)

  Note
  ----
  In general, the use of batch normalization can cause issues with NaNs. If
  you're having trouble with NaNs while using this model, consider setting
  `batch_normalize_kwargs={"trainable": False}` or turning off batch
  normalization entirely with `batch_normalize=False`.

  References
  ----------
  .. [1] Kearnes, Steven, et al. "Molecular graph convolutions: moving beyond 
         fingerprints." Journal of computer-aided molecular design 30.8 (2016): 
         595-608.

  """

  def __init__(self,
               n_tasks: int,
               n_atom_feat: OneOrMany[int] = 75,
               n_pair_feat: OneOrMany[int] = 14,
               n_hidden: int = 50,
               n_graph_feat: int = 128,
               n_weave: int = 2,
               fully_connected_layer_sizes: List[int] = [2000, 100],
               conv_weight_init_stddevs: OneOrMany[float] = 0.03,
               weight_init_stddevs: OneOrMany[float] = 0.01,
               bias_init_consts: OneOrMany[float] = 0.0,
               weight_decay_penalty: float = 0.0,
               weight_decay_penalty_type: str = "l2",
               dropouts: OneOrMany[float] = 0.25,
               final_conv_activation_fn: Optional[ActivationFn] = tf.nn.tanh,
               activation_fns: OneOrMany[ActivationFn] = tf.nn.relu,
               batch_normalize: bool = True,
               batch_normalize_kwargs: Dict = {
                   "renorm": True,
                   "fused": False
               },
               gaussian_expand: bool = True,
               compress_post_gaussian_expansion: bool = False,
               mode: str = "classification",
               n_classes: int = 2,
               batch_size: int = 100,
               **kwargs):
    """
    Parameters
    ----------
    n_tasks: int
      Number of tasks
    n_atom_feat: int, optional (default 75)
      Number of features per atom. Note this is 75 by default and should be 78
      if chirality is used by `WeaveFeaturizer`.
    n_pair_feat: int, optional (default 14)
      Number of features per pair of atoms.
    n_hidden: int, optional (default 50)
      Number of units(convolution depths) in corresponding hidden layer
    n_graph_feat: int, optional (default 128)
      Number of output features for each molecule(graph)
    n_weave: int, optional (default 2)
      The number of weave layers in this model.
    fully_connected_layer_sizes: list (default `[2000, 100]`)
      The size of each dense layer in the network.  The length of
      this list determines the number of layers.
    conv_weight_init_stddevs: list or float (default 0.03)
      The standard deviation of the distribution to use for weight
      initialization of each convolutional layer. The length of this lisst
      should equal `n_weave`. Alternatively, this may be a single value instead
      of a list, in which case the same value is used for each layer.
    weight_init_stddevs: list or float (default 0.01)
      The standard deviation of the distribution to use for weight
      initialization of each fully connected layer.  The length of this list
      should equal len(layer_sizes).  Alternatively this may be a single value
      instead of a list, in which case the same value is used for every layer.
    bias_init_consts: list or float (default 0.0)
      The value to initialize the biases in each fully connected layer.  The
      length of this list should equal len(layer_sizes).
      Alternatively this may be a single value instead of a list, in
      which case the same value is used for every layer.
    weight_decay_penalty: float (default 0.0)
      The magnitude of the weight decay penalty to use
    weight_decay_penalty_type: str (default "l2")
      The type of penalty to use for weight decay, either 'l1' or 'l2'
    dropouts: list or float (default 0.25)
      The dropout probablity to use for each fully connected layer.  The length of this list
      should equal len(layer_sizes).  Alternatively this may be a single value
      instead of a list, in which case the same value is used for every layer.
    final_conv_activation_fn: Optional[ActivationFn] (default `tf.nn.tanh`)
      The Tensorflow activation funcntion to apply to the final
      convolution at the end of the weave convolutions. If `None`, then no
      activate is applied (hence linear).
    activation_fns: list or object (default `tf.nn.relu`)
      The Tensorflow activation function to apply to each fully connected layer.  The length
      of this list should equal len(layer_sizes).  Alternatively this may be a
      single value instead of a list, in which case the same value is used for
      every layer.
    batch_normalize: bool, optional (default True)
      If this is turned on, apply batch normalization before applying
      activation functions on convolutional and fully connected layers.
    batch_normalize_kwargs: Dict, optional (default `{"renorm"=True, "fused": False}`)
      Batch normalization is a complex layer which has many potential
      argumentswhich change behavior. This layer accepts user-defined
      parameters which are passed to all `BatchNormalization` layers in
      `WeaveModel`, `WeaveLayer`, and `WeaveGather`.
    gaussian_expand: boolean, optional (default True)
      Whether to expand each dimension of atomic features by gaussian
      histogram
    compress_post_gaussian_expansion: bool, optional (default False)
      If True, compress the results of the Gaussian expansion back to the
      original dimensions of the input.
    mode: str (default "classification")
      Either "classification" or "regression" for type of model.
    n_classes: int (default 2)
      Number of classes to predict (only used in classification mode)
    batch_size: int (default 100)
      Batch size used by this model for training.
    """
    if mode not in ['classification', 'regression']:
      raise ValueError("mode must be either 'classification' or 'regression'")

    if not isinstance(n_atom_feat, SequenceCollection):
      n_atom_feat = [n_atom_feat] * n_weave
    if not isinstance(n_pair_feat, SequenceCollection):
      n_pair_feat = [n_pair_feat] * n_weave
    n_layers = len(fully_connected_layer_sizes)
    if not isinstance(conv_weight_init_stddevs, SequenceCollection):
      conv_weight_init_stddevs = [conv_weight_init_stddevs] * n_weave
    if not isinstance(weight_init_stddevs, SequenceCollection):
      weight_init_stddevs = [weight_init_stddevs] * n_layers
    if not isinstance(bias_init_consts, SequenceCollection):
      bias_init_consts = [bias_init_consts] * n_layers
    if not isinstance(dropouts, SequenceCollection):
      dropouts = [dropouts] * n_layers
    if not isinstance(activation_fns, SequenceCollection):
      activation_fns = [activation_fns] * n_layers
    if weight_decay_penalty != 0.0:
      if weight_decay_penalty_type == 'l1':
        regularizer = tf.keras.regularizers.l1(weight_decay_penalty)
      else:
        regularizer = tf.keras.regularizers.l2(weight_decay_penalty)
    else:
      regularizer = None

    self.n_tasks = n_tasks
    self.n_atom_feat = n_atom_feat
    self.n_pair_feat = n_pair_feat
    self.n_hidden = n_hidden
    self.n_graph_feat = n_graph_feat
    self.mode = mode
    self.n_classes = n_classes

    # Build the model.
    atom_features = Input(shape=(self.n_atom_feat[0],))
    pair_features = Input(shape=(self.n_pair_feat[0],))
    pair_split = Input(shape=tuple(), dtype=tf.int32)
    atom_split = Input(shape=tuple(), dtype=tf.int32)
    atom_to_pair = Input(shape=(2,), dtype=tf.int32)
    inputs = [atom_features, pair_features, pair_split, atom_to_pair]
    for ind in range(n_weave):
      n_atom = self.n_atom_feat[ind]
      n_pair = self.n_pair_feat[ind]
      if ind < n_weave - 1:
        n_atom_next = self.n_atom_feat[ind + 1]
        n_pair_next = self.n_pair_feat[ind + 1]
      else:
        n_atom_next = n_hidden
        n_pair_next = n_hidden
      weave_layer_ind_A, weave_layer_ind_P = layers.WeaveLayer(
          n_atom_input_feat=n_atom,
          n_pair_input_feat=n_pair,
          n_atom_output_feat=n_atom_next,
          n_pair_output_feat=n_pair_next,
          init=tf.keras.initializers.TruncatedNormal(
              stddev=conv_weight_init_stddevs[ind]),
          batch_normalize=batch_normalize)(inputs)
      inputs = [weave_layer_ind_A, weave_layer_ind_P, pair_split, atom_to_pair]
    # Final atom-layer convolution. Note this differs slightly from the paper
    # since we use a tanh activation as default. This seems necessary for numerical
    # stability.
    dense1 = Dense(self.n_graph_feat,
                   activation=final_conv_activation_fn)(weave_layer_ind_A)
    if batch_normalize:
      dense1 = BatchNormalization(**batch_normalize_kwargs)(dense1)
    weave_gather = layers.WeaveGather(
        batch_size,
        n_input=self.n_graph_feat,
        gaussian_expand=gaussian_expand,
        compress_post_gaussian_expansion=compress_post_gaussian_expansion)(
            [dense1, atom_split])

    if n_layers > 0:
      # Now fully connected layers
      input_layer = weave_gather
      for layer_size, weight_stddev, bias_const, dropout, activation_fn in zip(
          fully_connected_layer_sizes, weight_init_stddevs, bias_init_consts,
          dropouts, activation_fns):
        layer = Dense(
            layer_size,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(
                stddev=weight_stddev),
            bias_initializer=tf.constant_initializer(value=bias_const),
            kernel_regularizer=regularizer)(input_layer)
        if dropout > 0.0:
          layer = Dropout(rate=dropout)(layer)
        if batch_normalize:
          # Should this allow for training?
          layer = BatchNormalization(**batch_normalize_kwargs)(layer)
        layer = Activation(activation_fn)(layer)
        input_layer = layer
      output = input_layer
    else:
      output = weave_gather

    n_tasks = self.n_tasks
    if self.mode == 'classification':
      n_classes = self.n_classes
      logits = Reshape((n_tasks, n_classes))(Dense(n_tasks * n_classes)(output))
      output = Softmax()(logits)
      outputs = [output, logits]
      output_types = ['prediction', 'loss']
      loss: Loss = SoftmaxCrossEntropy()
    else:
      output = Dense(n_tasks)(output)
      outputs = [output]
      output_types = ['prediction']
      loss = L2Loss()
    model = tf.keras.Model(inputs=[
        atom_features, pair_features, pair_split, atom_split, atom_to_pair
    ],
                           outputs=outputs)
    super(EnhancedWeaveModel, self).__init__(model,
                                     loss,
                                     output_types=output_types,
                                     batch_size=batch_size,
                                     **kwargs)

  def compute_features_on_batch(self, X_b):
    """Compute tensors that will be input into the model from featurized representation.

    The featurized input to `WeaveModel` is instances of `WeaveMol` created by
    `WeaveFeaturizer`. This method converts input `WeaveMol` objects into
    tensors used by the Keras implementation to compute `WeaveModel` outputs.

    Parameters
    ----------
    X_b: np.ndarray
      A numpy array with dtype=object where elements are `WeaveMol` objects.

    Returns
    -------
    atom_feat: np.ndarray
      Of shape `(N_atoms, N_atom_feat)`.
    pair_feat: np.ndarray
      Of shape `(N_pairs, N_pair_feat)`. Note that `N_pairs` will depend on
      the number of pairs being considered. If `max_pair_distance` is
      `None`, then this will be `N_atoms**2`. Else it will be the number
      of pairs within the specifed graph distance.
    pair_split: np.ndarray
      Of shape `(N_pairs,)`. The i-th entry in this array will tell you the
      originating atom for this pair (the "source"). Note that pairs are
      symmetric so for a pair `(a, b)`, both `a` and `b` will separately be
      sources at different points in this array.
    atom_split: np.ndarray
      Of shape `(N_atoms,)`. The i-th entry in this array will be the molecule
      with the i-th atom belongs to.
    atom_to_pair: np.ndarray
      Of shape `(N_pairs, 2)`. The i-th row in this array will be the array
      `[a, b]` if `(a, b)` is a pair to be considered. (Note by symmetry, this
      implies some other row will contain `[b, a]`.
    """
    atom_feat = []
    pair_feat = []
    atom_split = []
    atom_to_pair = []
    pair_split = []
    additional_info = []
    start = 0
    for im, mol in enumerate(X_b):
      n_atoms = mol.get_num_atoms()
      # pair_edges is of shape (2, N)
      pair_edges = mol.get_pair_edges()
      N_pairs = pair_edges[1]
      # number of atoms in each molecule
      atom_split.extend([im] * n_atoms)
      # index of pair features
      C0, C1 = np.meshgrid(np.arange(n_atoms), np.arange(n_atoms))
      atom_to_pair.append(pair_edges.T + start)
      # Get starting pair atoms
      pair_starts = pair_edges.T[:, 0]
      # number of pairs for each atom
      pair_split.extend(pair_starts + start)
      start = start + n_atoms

      # atom features
      atom_feat.append(mol.get_atom_features())
      # pair features
      pair_feat.append(mol.get_pair_features())
      
      # get additional info
      additional_info.append(mol.get_additional_info())

    return (np.concatenate(atom_feat, axis=0), np.concatenate(pair_feat,
                                                              axis=0),
            np.array(pair_split), np.array(atom_split),
            np.concatenate(atom_to_pair, axis=0),
            np.array(additional_info))

  def default_generator(
      self,
      dataset: Dataset,
      epochs: int = 1,
      mode: str = 'fit',
      deterministic: bool = True,
      pad_batches: bool = True) -> Iterable[Tuple[List, List, List]]:
    """Convert a dataset into the tensors needed for learning.

    Parameters
    ----------
    dataset: `dc.data.Dataset`
      Dataset to convert
    epochs: int, optional (Default 1)
      Number of times to walk over `dataset`
    mode: str, optional (Default 'fit')
      Ignored in this implementation.
    deterministic: bool, optional (Default True)
      Whether the dataset should be walked in a deterministic fashion
    pad_batches: bool, optional (Default True)
      If true, each returned batch will have size `self.batch_size`.

    Returns
    -------
    Iterator which walks over the batches
    """

    for epoch in range(epochs):
      for (X_b, y_b, w_b,
           ids_b) in dataset.iterbatches(batch_size=self.batch_size,
                                         deterministic=deterministic,
                                         pad_batches=pad_batches):
        if y_b is not None:
          if self.mode == 'classification':
            y_b = to_one_hot(y_b.flatten(),
                             self.n_classes).reshape(-1, self.n_tasks,
                                                     self.n_classes)
        inputs = self.compute_features_on_batch(X_b)
        yield (inputs, [y_b], [w_b])

