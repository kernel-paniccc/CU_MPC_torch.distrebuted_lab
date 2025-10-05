import time
import logging
import torch
import crypten
import crypten.communicator as comm
import crypten.nn as nn
from crypten.nn.loss import MSELoss
from crypten.optim import SGD

from config import config

from . import task

def init():
    crypten.init()
    comm.get().set_verbosity(True)
    crypten.encoder.set_default_precision(config.PRECISION)

@task("ttp")
def ttp():
    crypten.mpc.provider.TTPServer()


@task("linreg")
def linreg():
    init()

    rank = comm.get().get_rank()

    x_train_enc1 = crypten.load_from_party(config.X_TRAIN_PATH_WORK1, src=0)
    y_train_enc1 = crypten.load_from_party(config.Y_TRAIN_PATH_WORK1, src=0)
    x_train_enc2 = crypten.load_from_party(config.X_TRAIN_PATH_WORK2, src=1)
    y_train_enc2 = crypten.load_from_party(config.Y_TRAIN_PATH_WORK2, src=1)
    x_test_enc = crypten.load_from_party(config.X_TEST_PATH, src=0)
    y_test_enc = crypten.load_from_party(config.Y_TEST_PATH, src=0)


    X_train_enc = crypten.cat([x_train_enc1, x_train_enc2], dim=0)
    y_train_enc = crypten.cat([y_train_enc1, y_train_enc2], dim=0)

    # Модель
    class LinearModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, 1)

        def forward(self, x):
            return self.fc(x)

    input_dim = X_train_enc.shape[1]
    model = LinearModel(input_dim)

    # инициализируем веса
    for name, weight in model.named_parameters():
        if 'weight' in name:
            nn.init.normal_(weight, mean=0.0, std=0.01)
        elif 'bias' in name:
            nn.init.constant_(weight, 0.)
    model.encrypt()  # переводим веса модели в разделения секрета

    # Обучение
    criterion = MSELoss()
    optimizer = SGD(model.parameters(), lr=0.05, momentum=0.9)

    batch_size = config.BATCH_SIZE
    n_epochs = config.EPOCHS

    n_samples = y_train_enc.size(0)
    n_batches = (n_samples + batch_size - 1) // batch_size

    t0 = time.time()

    comm.get().reset_communication_stats()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for i in range(n_batches):
            start, end = i * batch_size, min((i + 1) * batch_size, n_samples)

            X_batch = X_train_enc[start:end]
            y_batch = y_train_enc[start:end]

            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.get_plain_text().item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logging.info(f"[Epoch {epoch + 1:02d}] loss={epoch_loss / n_batches:.6f}")

    logging.info(f"Model train elapsed time: {time.time() - t0:.4f} seconds")
    comm.get().print_communication_stats()
    # Валидация на X_test
    with torch.no_grad():
        y_pred_enc = model(x_test_enc)
        test_loss = criterion(y_pred_enc, y_test_enc)
        logging.info(f"Test MSE: {test_loss.get_plain_text().item()}")

    # Веса модели
    w_learned_enc = model.fc.weight
    b_learned_enc = model.fc.bias

    logging.info(f"Learned weights share: {w_learned_enc.share.view(-1)}")
    logging.info(f"Learned bias share: {b_learned_enc.share.item()}")
    logging.info(f"End worker with rank: {rank}")

    crypten.uninit()