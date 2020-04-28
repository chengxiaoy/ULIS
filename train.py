from train_new import train_epoch_group, buildConfig

if __name__ == '__main__':
    config = buildConfig(0)
    config.expriment_id = 1
    config.loss = 'focal'
    train_epoch_group(config)

    config = buildConfig(0)
    config.expriment_id = 2
    config.loss = 'ce'
    train_epoch_group(config)

    config = buildConfig(0)
    config.expriment_id = 3
    config.loss = 'focal'
    config.schedular = 'cos'
    train_epoch_group(config)

    config = buildConfig(0)
    config.expriment_id = 4
    config.loss = 'focal'
    config.schedular = 'cyc'
    train_epoch_group(config)
