import os
from collections import namedtuple
import tensorflow as tf

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


class SenseBert:
    def __init__(self, name_or_path, max_seq_length=512):
        self.max_seq_length = max_seq_length
        self.model, self.serving_fn = _load_model(name_or_path)
        self.tokenizer = load_tokenizer(name_or_path)

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

    def run(self, input_ids, input_mask):
        """
        Run inference on the model.
        
        Args:
            input_ids: List or array of input token IDs
            input_mask: List or array of attention masks
            
        Returns:
            Tuple of (contextualized_embeddings, mlm_logits, supersense_logits)
        """
        # Convert to tensors
        input_ids_tensor = tf.constant(input_ids, dtype=tf.int32)
        input_mask_tensor = tf.constant(input_mask, dtype=tf.int32)
        
        # Run the model
        outputs = self.serving_fn(
            input_ids=input_ids_tensor,
            input_mask=input_mask_tensor
        )
        
        # Extract outputs based on available keys
        contextualized_embeddings = None
        mlm_logits = None
        supersense_logits = None
        
        # Handle different output formats
        if isinstance(outputs, dict):
            # Try to get contextualized embeddings
            if 'contextualized_embeddings' in outputs:
                contextualized_embeddings = outputs['contextualized_embeddings']
            
            # Get MLM logits
            if 'masked_lm' in outputs:
                mlm_logits = outputs['masked_lm']
            elif 'mlm_logits' in outputs:
                mlm_logits = outputs['mlm_logits']
            
            # Get supersense logits
            if 'ss' in outputs:
                supersense_logits = outputs['ss']
            elif 'supersense_logits' in outputs:
                supersense_logits = outputs['supersense_logits']
        
        return contextualized_embeddings, mlm_logits, supersense_logits

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