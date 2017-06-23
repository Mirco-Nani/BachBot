parameters = [
    ("exp02" , {
        "target" : "complex_rnn",

        # Parameters
        "learning_rate" : 0.001,
        "training_iters" : 15000*128,
        "batch_size" : 128,
        "display_step" : 10,
        "n_input_symbols" : dataset.get_vocabulary_size(),
        "n_output_symbols" : dataset.get_vocabulary_size(),
        "embedding_size" : 128,
        "use_gpu" : True,
        "full_gpu": True,

        "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
        #"save_location" : os.path.join(experiment_path,"checkpoints/"),#"../../experiments/midi/bach/exp02/checkpoints/",
        #"tensorboard_location" : os.path.join(experiment_path,"tensorboard/"),#"../../experiments/midi/bach/exp02/tensorboard/",
        "start_checkpoint" : 5000,

        # Network Parameters
        "dropout_input_keep_prob" : 1.0,
        "dropout_output_keep_prob" : 1.0,
        "feed_previous" : True,
        "layers" : 2, # more layers means slower training time (obviously)
        "attention" : False, # attention makes training veeery slow 

        "encoder_size" : 64,
        "decoder_size" : 64,
        "n_hidden" : 512, # hidden layer num of features
        #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
        "n_classes" : dataset.get_vocabulary_size()
    }),
    ("exp07" , {
        "target" : "complex_rnn",

        # Parameters
        "learning_rate" : 0.001,
        "training_iters" : 5000*128,
        "batch_size" : 128,
        "display_step" : 10,
        "n_input_symbols" : dataset.get_vocabulary_size(),
        "n_output_symbols" : dataset.get_vocabulary_size(),
        "embedding_size" : 128,
        "use_gpu" : True,
        "full_gpu": True,

        "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
        #"save_location" : os.path.join(experiment_path,"checkpoints/"),#"../../experiments/midi/bach/exp02/checkpoints/",
        #"tensorboard_location" : os.path.join(experiment_path,"tensorboard/"),#"../../experiments/midi/bach/exp02/tensorboard/",
        "start_checkpoint" : 0,

        # Network Parameters
        "dropout_input_keep_prob" : 1.0,
        "dropout_output_keep_prob" : 1.0,
        "feed_previous" : True,
        "layers" : 4, # more layers means slower training time (obviously)
        "attention" : True, # attention makes training veeery slow 

        "encoder_size" : 256,
        "decoder_size" : 256,
        "n_hidden" : 512, # hidden layer num of features
        #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
        "n_classes" : dataset.get_vocabulary_size()
    }),
    ("exp03" , {
        "target" : "complex_rnn",

        # Parameters
        "learning_rate" : 0.001,
        "training_iters" : 5000*128,
        "batch_size" : 128,
        "display_step" : 10,
        "n_input_symbols" : dataset.get_vocabulary_size(),
        "n_output_symbols" : dataset.get_vocabulary_size(),
        "embedding_size" : 128,
        "use_gpu" : True,
        "full_gpu": True,

        "save_step" : 250, # WARNING! saving takes a LOOOOOOT of time!
        #"save_location" : os.path.join(experiment_path,"checkpoints/"),#"../../experiments/midi/bach/exp02/checkpoints/",
        #"tensorboard_location" : os.path.join(experiment_path,"tensorboard/"),#"../../experiments/midi/bach/exp02/tensorboard/",
        "start_checkpoint" : 0,

        # Network Parameters
        "dropout_input_keep_prob" : 1.0,
        "dropout_output_keep_prob" : 1.0,
        "feed_previous" : True,
        "layers" : 4, # more layers means slower training time (obviously)
        "attention" : False, # attention makes training veeery slow 

        "encoder_size" : 256,
        "decoder_size" : 256,
        "n_hidden" : 512, # hidden layer num of features
        #"n_classes" : sequence_length, # (0 to sequence_length-1 digits)
        "n_classes" : dataset.get_vocabulary_size()
    }),
    
]