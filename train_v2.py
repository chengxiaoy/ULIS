from train_new import train_epoch_group, buildConfig,test_config

if __name__ == '__main__':

    config_list = []
    config = buildConfig(1)
    config.expriment_id = 101
    config.loss = 'focal'
    config.use_swa = True
    config_list.append(config)

    config = buildConfig(1)
    config.expriment_id = 102
    config.loss = 'ce'
    config.use_swa = True
    config_list.append(config)


    config = buildConfig(1)
    config.expriment_id = 103
    config.loss = 'focal'
    config.schedular = 'cos'
    config.use_swa = True
    config_list.append(config)


    config = buildConfig(1)
    config.expriment_id = 104
    config.loss = 'focal'
    config.schedular = 'cyc'
    config.use_swa = True
    config_list.append(config)

    for con in config_list:
        test_config(con)


    for con in config_list:
        train_epoch_group(con)
