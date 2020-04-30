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

    config = buildConfig(0)
    config.expriment_id = 5
    config.loss = 'ce'
    config.schedular = 'cyc'
    config_list.append(config)

    config = buildConfig(0)
    config.expriment_id = 6
    config.loss = 'focal'
    config.schedular = 'cyc'
    config.model_name = 'seq2seq'
    config_list.append(config)


    # for con in config_list:
    #     test_config(con)

    for con in config_list:
        train_epoch_group(con)
