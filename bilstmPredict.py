import bilstmTrain

if __name__ == "__main__": 
    flavor = sys.argv[1]
    model_file = sys.argv[2]
    test_file = sys.argv[3]
    tagging_type = sys.argv[4]
   
    RUN_PARAMS = bilstmTrain.FAVORITE_RUN_PARAMS
    RUN_PARAMS.update(
                { 'FLAVOR': flavor, 
                'TRAIN_FILE': None,
                'DEV_FILE' : None, #dev_file,
                'TAGGING_TYPE' : tagging_type,
                'TEST_FILE': test_file,
                'TEST_O_FILE': tagging_type + "test_predictions", #test_o_file,
                'MODEL_FILE': model_file,
                'SAVE_TO_FILE': False, 
                'RUN_DEV' : False, 
                'DROPOUT' : True}
    )

    run = bilstmTrain.Run(RUN_PARAMS)

    run.test()
