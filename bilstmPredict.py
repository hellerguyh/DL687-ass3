import bilstmTrain

if __name__ == "__main__": 
    flavor = sys.argv[1]
    model_file = sys.argv[2]
    test_file = sys.argv[3]
    tagging_type = sys.argv[4]
    
    run = Run({ 'FLAVOR': flavor, 
                'EMBEDDING_DIM' : 50, 
                'RNN_H_DIM' : 50, 
                'EPOCHS' : 5, 
                'BATCH_SIZE' : 100, 
                'CHAR_EMBEDDING_DIM': 30, 
                'TRAIN_FILE': None,
                'DEV_FILE' : None, #dev_file,
                'TAGGING_TYPE' : tagging_type,
                'TEST_FILE': test_file,
                'TEST_O_FILE': tagging_type + "test_predictions", #test_o_file,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': False, 
                'RUN_DEV' : False})

    run.test()
