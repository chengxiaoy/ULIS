from train_new import train_epoch_group, buildConfig, test_config

if __name__ == '__main__':

    config_list = []

    # config = buildConfig(0)
    # config.expriment_id = 1
    # config.loss = 'focal'
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 2
    # config.loss = 'ce'
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 3
    # config.loss = 'focal'
    # config.schedular = 'cos'
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 4
    # config.loss = 'focal'
    # config.schedular = 'cyc'
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 5
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 6
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.model_name = 'seq2seq'
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 7
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 8
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True

    # config = buildConfig(0)
    # config.expriment_id = 9
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True

    # config = buildConfig(0)
    # config.NNBATCHSIZE = 16
    # config.expriment_id = 10
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True

    # config = buildConfig(0)
    # config.expriment_id = 80
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 81
    # config.GROUP_BATCH_SIZE=2000
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 82
    # config.NNBATCHSIZE=16
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = False
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(1)
    # config.expriment_id = 83
    # config.data_fe = 'shifted'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 84
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.use_se = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 85
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.use_se = False
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 86
    # config.data_fe = 'shifted'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.use_se = False
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 87
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 88
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 89
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)

    # config = buildConfig(0)
    # config.NNBATCHSIZE = 64
    # config.expriment_id = 90
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.GROUP_BATCH_SIZE = 8000
    # config.expriment_id = 91
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config_list.append(config)
    #

    # config = buildConfig(0)
    # config.expriment_id = 92
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.gaussian_noise = True
    # config_list.append(config)
    #
    # config = buildConfig(0)
    # config.expriment_id = 93
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.5
    # config_list.append(config)



    # config = buildConfig(0)
    # config.data_type  = 'clean'
    # config.expriment_id = 94
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.5
    # config_list.append(config)



    # shift 1,2,3
    # config = buildConfig(0)
    # config.expriment_id = 95
    # config.data_fe = 'shifted_viterbi_proba'
    #
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.5
    # config_list.append(config)

    # config = buildConfig(0)
    # config.expriment_id = 99
    # config.model_name = 'unet'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.2
    # config_list.append(config)



    # config = buildConfig(0)
    # config.expriment_id = 1100
    # config.NNBATCHSIZE = 64
    # config.GROUP_BATCH_SIZE = 2000
    # config.data_fe = 'shifted_mix_proba'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.2
    # config_list.append(config)
    #
    #
    #
    # config = buildConfig(0)
    # config.expriment_id = 1101
    # config.NNBATCHSIZE = 16
    # config.GROUP_BATCH_SIZE = 8000
    # config.data_fe = 'shifted_mix_proba'
    # config.loss = 'ce'
    # config.schedular = 'cyc'
    # config.early_stop_max = True
    # config.use_cbr = True
    # config.drop_out = 0.2
    # config_list.append(config)


    config = buildConfig(0)
    config.expriment_id = 155
    config.data_fe = 'shifted_viterbi_proba'
    config.loss = 'ce'
    config.schedular = 'cyc'
    config.early_stop_max = True
    config.use_cbr = True
    config.drop_out = 0.5
    config.residual = True
    config_list.append(config)

    # for con in config_list:
    #     test_config(con)

    for con in config_list:
        train_epoch_group(con)
