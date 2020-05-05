from train_new import train_epoch_group, buildConfig,test_config

if __name__ == '__main__':

    config_list = []
    # config = buildConfig(1)
    # config.expriment_id = 101
    # config.loss = 'focal'
    # config.use_swa = True
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.expriment_id = 102
    # config.loss = 'ce'
    # config.use_swa = True
    # config_list.append(config)
    #
    #
    # config = buildConfig(1)
    # config.expriment_id = 103
    # config.loss = 'focal'
    # config.schedular = 'cos'
    # config.use_swa = True
    # config_list.append(config)


    # config = buildConfig(1)
    # config.expriment_id = 104
    # config.loss = 'focal'
    # config.schedular = 'cyc'
    # config.use_swa = True
    # config_list.append(config)


    # config = buildConfig(1)
    # config.expriment_id = 105
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.use_swa = True
    # config_list.append(config)




    # config = buildConfig(1)
    # config.expriment_id = 107
    # config.loss = 'focal'
    # config.schedular = 'cyc'
    # config.data_type = 'clean'
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 108
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.data_type = 'clean'
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 109
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.use_cbr = True
    # config.data_type = 'clean'
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 11
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config.group_train = True
    #
    # config_list.append(config)


    # config = buildConfig(1)
    # config.expriment_id = 12
    # config.GROUP_BATCH_SIZE = 800
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config.group_train = True

    # config = buildConfig(0)
    # config.expriment_id = 110
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 111
    # config.NNBATCHSIZE = 16
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 112
    # config.GROUP_BATCH_SIZE = 2000
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 31
    # # config.GROUP_BATCH_SIZE=2000
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.expriment_id = 32
    # # config.NNBATCHSIZE=16
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.expriment_id = 113
    # config.data_fe = 'shifted'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)
    #
    #
    # config = buildConfig(1)
    # config.expriment_id = 114
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.expriment_id = 115
    # config.NNBATCHSIZE = 16
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.group_train = True
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 120
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(1)
    # config.NNBATCHSIZE = 16
    # config.expriment_id = 121
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.GROUP_BATCH_SIZE = 2000
    # config.expriment_id = 122
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    config = buildConfig(1)
    config.expriment_id = 123
    config.loss = 'ce'
    config.schedular = 'cyc'
    config.early_stop_max = True
    config.use_cbr = True
    config.gaussian_noise = True
    config_list.append(config)

    config = buildConfig(1)
    config.expriment_id = 124
    config.loss = 'ce'
    config.schedular = 'cyc'
    config.early_stop_max = True
    config.use_cbr = True
    config.drop_out = 0.5
    config_list.append(config)







    # for con in config_list:
    #     test_config(con)
    # config = buildConfig(1)
    # config.expriment_id = 105
    # config.loss = 'focal'
    # config.use_swa = True
    # config.model_name = 'seq2seq'
    #
    # config_list.append(config)
    #
    # config = buildConfig(1)
    # config.expriment_id = 106
    # config.loss = 'ce'
    # config.use_swa = True
    # config.model_name = 'seq2seq'
    #
    # config_list.append(config)
    #
    #
    # config = buildConfig(1)
    # config.expriment_id = 107
    # config.loss = 'focal'
    # config.schedular = 'cos'
    # config.use_swa = True
    # config.model_name = 'seq2seq'
    #
    # config_list.append(config)



    for con in config_list:
        train_epoch_group(con)
