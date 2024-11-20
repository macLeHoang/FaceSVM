class Config(object):
    env = 'default'
    backbone = 'resnet18'
    classify = 'softmax'
    num_classes = 10575
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    use_hingle_loss = True
    delta = 5
    hingle_weight = 0.5

    display = False
    finetune = False

    train_root = '/content/CASIA-maxpy-clean'
    train_list = '/data/Datasets/webface/train_data_13938.txt'
    val_list = '/data/Datasets/webface/val_data_13938.txt'

    test_root = '/data1/Datasets/anti-spoofing/test/data_align_256'
    test_list = 'test.txt'

    lfw_root = 'lfw-align-128/lfw-align-128'
    lfw_test_list = 'lfw_test_pair.txt'

    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet18.pth'
    test_model_path = 'resnet18_last_CE_SVM_0.1.pth'
    save_interval = 1

    train_batch_size = 32  # batch size
    test_batch_size = 60

    input_shape = (1, 128, 128)

    use_gpu = True  # use GPU or not
    gpu_id = '0'
    num_workers = 2  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50

    optimizer = 'sgd'
    lr = 1e-1  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4

    resume="resnet18_last.pth" # str, path to resume weight