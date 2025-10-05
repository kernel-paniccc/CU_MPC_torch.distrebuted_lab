import os


class Config:
    # Логирование
    LOGLEVEL = os.getenv("LOGLEVEL", "INFO")
    X_TEST_PATH = os.getenv("X_TEST_PATH", "/data/linreg/x_test_norm_worker1.pth")
    Y_TEST_PATH = os.getenv("Y_TEST_PATH", "/data/linreg/y_test_worker1.pth")
    X_TRAIN_PATH_WORK1 = os.getenv("X_TRAIN_PATH_W1", "/data/linreg/x_train_norm_worker1.pth")
    X_TRAIN_PATH_WORK2 = os.getenv("X_TRAIN_PATH_W2", "/data/linreg/x_train_norm_worker2.pth")
    Y_TRAIN_PATH_WORK1 = os.getenv("Y_TRAIN_PATH_W1", "/data/linreg/y_train_worker1.pth")
    Y_TRAIN_PATH_WORK2 = os.getenv("Y_TRAIN_PATH_W2", "/data/linreg/y_train_worker2.pth")
    PRECISION = int(os.getenv("PRECISION", 16))
    EPOCHS = int(os.getenv("EPOCHS", 20))
    BATCH_SIZE = int(os.getenv("BATCH_SIZE", 512))



config = Config()
