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

    config = buildConfig(0)
    config.expriment_id = 87
    config.loss = 'ce'
    config.schedular = 'cyc'
    config.early_stop_max = True
    config.use_cbr = True
    config_list.append(config)

    # for con in config_list:
    #     test_config(con)

    for con in config_list:
        train_epoch_group(con)
