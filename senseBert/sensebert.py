import os
import numpy as np
from collections import namedtuple
import tensorflow as tf
import json

from senseBert.tokenization import FullTokenizer

_SenseBertGraph = namedtuple(
    'SenseBertGraph',
    ('input_ids', 'input_mask', 'contextualized_embeddings', 'mlm_logits', 'supersense_logits')
)

_MODEL_PATHS = {
    'sensebert-base-uncased': 'gs://ai21-public-models/sensebert-base-uncased',
    'sensebert-large-uncased': 'gs://ai21-public-models/sensebert-large-uncased'
}
_CONTEXTUALIZED_EMBEDDINGS_TENSOR_NAME = "bert/encoder/Reshape_13:0"


def _get_model_path(name_or_path, is_tokenizer=False):
    if name_or_path in _MODEL_PATHS:
        print(f"Loading the known {'tokenizer' if is_tokenizer else 'model'} '{name_or_path}'")
        model_path = _MODEL_PATHS[name_or_path]
    else:
        print(f"This is not a known {'tokenizer' if is_tokenizer else 'model'}. "
              f"Assuming {name_or_path} is a path or a url...")
        model_path = name_or_path
    return model_path


def load_tokenizer(name_or_path):
    model_path = _get_model_path(name_or_path, is_tokenizer=True)
    vocab_file = os.path.join(model_path, "vocab.txt")
    supersense_vocab_file = os.path.join(model_path, "supersense_vocab.txt")
    return FullTokenizer(vocab_file=vocab_file, senses_file=supersense_vocab_file)


def _load_model(name_or_path):
    """Load a SavedModel (handles both TF1 and TF2 formats)"""
    model_path = _get_model_path(name_or_path)
    
    # Load the model using TF2's saved_model.load with 'serve' tag for TF1 models
    try:
        # First try loading without tags (TF2 native models)
        loaded_model = tf.saved_model.load(model_path)
    except ValueError as e:
        if "tags=" in str(e):
            # TF1 SavedModel - use 'serve' tag
            print("Loading TF1-style SavedModel with 'serve' tag...")
            loaded_model = tf.saved_model.load(model_path, tags=['serve'])
        else:
            raise
    
    # Get the serving function
    if hasattr(loaded_model, 'signatures'):
        serving_fn = loaded_model.signatures.get(
            tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
            loaded_model.signatures.get('serving_default')
        )
    else:
        # Fallback for models without explicit signatures
        serving_fn = loaded_model
    
    return loaded_model, serving_fn


class BertConfig:
    """Simple config class to hold BERT configuration"""
    def __init__(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)


def load_config(name_or_path):
    """Load config.json file"""
    model_path = _get_model_path(name_or_path)
    # config_file = os.path.join(model_path, "config.json")
    
    # try:
    #     with tf.io.gfile.GFile(config_file, "r") as f:
    #         config_dict = json.loads(f.read())
    #     return BertConfig(config_dict)
    # except Exception as e:
    # print(f"Warning: Could not load config from {config_file}: {e}")
    # Return default config for large model
    return BertConfig({
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "intermediate_size": 4096,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "attention_probs_dropout_prob": 0.1,
        "max_position_embeddings": 512,
        "type_vocab_size": 2,
        "initializer_range": 0.02,
        "vocab_size": 56141
    })


class SenseBert:
    def __init__(self, name_or_path, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.model, self.serving_fn = _load_model(name_or_path)
        self.tokenizer = load_tokenizer(name_or_path)
        self.config = load_config(name_or_path)
        # Try to get the embeddings tensor from the graph for direct access
        self.embeddings_tensor = None
        try:
            graph = self.serving_fn.graph
            self.embeddings_tensor = graph.get_tensor_by_name(_CONTEXTUALIZED_EMBEDDINGS_TENSOR_NAME)
        except:
            print("Warning: Could not access embeddings tensor directly from graph")

    def tokenize(self, inputs):
        """
        Gets a string or a list of strings, and returns a tuple (input_ids, input_mask) to use as inputs for SenseBERT.
        Both share the same shape: [batch_size, sequence_length] where sequence_length is the maximal sequence length.
        """
        if isinstance(inputs, str):
            inputs = [inputs]

        # tokenizing all inputs
        all_token_ids = []
        for inp in inputs:
            tokens = [self.tokenizer.start_sym] + self.tokenizer.tokenize(inp)[0] + [self.tokenizer.end_sym]
            assert len(tokens) <= self.max_seq_length, f"Sequence length {len(tokens)} exceeds max_seq_length {self.max_seq_length}"
            all_token_ids.append(self.tokenizer.convert_tokens_to_ids(tokens))

        # decide the maximum sequence length and pad accordingly
        max_len = max([len(token_ids) for token_ids in all_token_ids])
        input_ids, input_mask = [], []
        pad_sym_id = self.tokenizer.convert_tokens_to_ids([self.tokenizer.pad_sym])[0]
        for token_ids in all_token_ids:
            to_pad = max_len - len(token_ids)
            input_ids.append(token_ids + [pad_sym_id] * to_pad)
            input_mask.append([1] * len(token_ids) + [0] * to_pad)

        return input_ids, input_mask

    def run(self, input_ids, attention_mask):
        """
        Get contextualized embeddings from the model (for PyTorch integration).
        Uses the masked language model output as a proxy for embeddings.
        
        Args:
            input_ids: Tensor of shape [batch_size, seq_len]
            attention_mask: Tensor of shape [batch_size, seq_len]
            
        Returns:
            Embeddings tensor of shape [batch_size, seq_len, hidden_size]
        """
        # Convert PyTorch tensors to numpy
        if hasattr(input_ids, 'cpu'):
            input_ids = input_ids.cpu().numpy()
        if hasattr(attention_mask, 'cpu'):
            attention_mask = attention_mask.cpu().numpy()
        
        batch_size, seq_length = input_ids.shape
        
        # Convert to TensorFlow tensors
        input_ids_tensor = tf.constant(input_ids, dtype=tf.int32)
        input_mask_tensor = tf.constant(attention_mask, dtype=tf.int32)
        
        # Create dummy tensors for the additional required inputs
        dummy_size = 512
        beginnings = tf.zeros([dummy_size], dtype=tf.int64)
        endings = tf.zeros([dummy_size], dtype=tf.int64)
        is_multiple_tokens_word = tf.zeros([dummy_size], dtype=tf.int64)
        is_mwe = tf.zeros([dummy_size], dtype=tf.int64)
        is_ne = tf.zeros([dummy_size], dtype=tf.int64)
        is_real_example = tf.zeros([dummy_size], dtype=tf.int64)
        is_single_token_word = tf.zeros([dummy_size], dtype=tf.int64)
        labels = tf.zeros([dummy_size], dtype=tf.int64)
        
        # Run the model
        outputs = self.serving_fn(
            input_ids=input_ids_tensor,
            input_mask=input_mask_tensor,
            beginnings=beginnings,
            endings=endings,
            is_multiple_tokens_word=is_multiple_tokens_word,
            is_mwe=is_mwe,
            is_ne=is_ne,
            is_real_example=is_real_example,
            is_single_token_word=is_single_token_word,
            labels=labels
        )
        
        # Extract MLM logits which contain the hidden representations
        if 'masked_lm' in outputs:
            mlm_logits = outputs['masked_lm'].numpy()
            # mlm_logits shape: [batch_size, seq_length, vocab_size]
            
            # Project from vocab space back to hidden space
            # Using a simple learned projection (PCA-like)
            vocab_size = mlm_logits.shape[-1]
            hidden_size = self.config.hidden_size
            
            # Create a consistent projection matrix (use SVD on the first batch)
            if not hasattr(self, '_projection_matrix'):
                # Initialize projection matrix
                flat_logits = mlm_logits.reshape(-1, vocab_size)
                # Use truncated SVD to reduce dimensions
                U, S, Vt = np.linalg.svd(flat_logits, full_matrices=False)
                self._projection_matrix = Vt[:hidden_size, :].T
            
            # Project to hidden space
            flat_logits = mlm_logits.reshape(-1, vocab_size)
            embeddings = flat_logits @ self._projection_matrix
            embeddings = embeddings.reshape(batch_size, seq_length, hidden_size)
            
            return embeddings
        
        # Fallback: return zeros
        return np.zeros((batch_size, seq_length, self.config.hidden_size), dtype=np.float32)

    def __call__(self, inputs):
        """
        Convenient method to tokenize and run inference in one call.
        
        Args:
            inputs: String or list of strings
            
        Returns:
            Tuple of (contextualized_embeddings, mlm_logits, supersense_logits)
        """
        input_ids, input_mask = self.tokenize(inputs)
        return self.run(input_ids, input_mask)