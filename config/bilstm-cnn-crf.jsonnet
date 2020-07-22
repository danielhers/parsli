local batch_size = 10;
local char_embedding_dim = 30;
local cnn_windows = [3];
local cnn_num_filters = 30;
local cuda_device = 0;
local embedding_dim = 100;
local dropout = 0.5;
local lstm_hidden_size = 200;
local lr = 0.015;
local num_epochs = 150;
local optimizer = "sgd";
local pretrained_embedding_file = "https://allennlp.s3.amazonaws.com/datasets/glove/glove.6B.100d.txt.gz";

{
  "dataset_reader": {
    "type": "conll2003",
    "tag_label": "ner",
    "coding_scheme": "BIOUL",
    "token_indexers": {
      "tokens": {
        "type": "single_id",
        "lowercase_tokens": true
      },
      "token_characters": {
        "type": "characters",
        "min_padding_length": 3
      }
    }
  },
  "datasets_for_vocab_creation": ["train"],
  "train_data_path": "./data/eng.train",
  "validation_data_path": "./data/eng.testa",
  "test_data_path": "./data/eng.testb",
  "evaluate_on_test": true,
  "model": {
    "type": "crf_tagger",
    "label_encoding": "BIOUL",
    "constrain_crf_decoding": true,
    "calculate_span_f1": true,
    "dropout": dropout,
    "include_start_end_transitions": false,
    "text_field_embedder": {
      "token_embedders": {
        "tokens": {
            "type": "embedding",
            "embedding_dim": embedding_dim,
            "pretrained_file": pretrained_embedding_file,
            "trainable": true
        },
        "token_characters": {
            "type": "character_encoding",
            "embedding": {
                "embedding_dim": char_embedding_dim,
            },
            "encoder": {
                "type": "cnn",
                "embedding_dim": char_embedding_dim,
                "num_filters": cnn_num_filters,
                "ngram_filter_sizes": cnn_windows,
                "conv_layer_activation": "relu"
            }
          }
       },
    },
    "encoder": {
        "type": "lstm",
        "input_size": embedding_dim + cnn_num_filters,
        "hidden_size": lstm_hidden_size,
        "dropout": dropout,
        "bidirectional": true
    },
  },
  "data_loader": {
    "batch_size": batch_size,
  },
  "trainer": {
    "optimizer": {
      "type": optimizer,
      "lr": lr,
    },
    "checkpointer": {
        "num_serialized_models_to_keep": 3,
    },
    "validation_metric": "+f1-measure-overall",
    "num_epochs": num_epochs,
    "patience": 25,
    "cuda_device": cuda_device
  }
}
