from torch.optim.optimizer import Optimizer
import matplotlib.pyplot as plt
from typing import Any, Dict 
from pathlib import Path
import torch.nn as nn
import torch.optim
import torch


WEIGHT: float = 0.7
BAIS: float = 0.3

start: int = 0
end: int = 1
step: float = 0.02

X: torch.Tensor = torch.arange(start, end, step).unsqueeze(dim=1)
y: torch.Tensor = WEIGHT * X + BAIS

TRAIN_TEST_split: int = int(0.8 * len(X))
X_train, y_train = X[:TRAIN_TEST_split], y[:TRAIN_TEST_split]
X_test, y_test = X[TRAIN_TEST_split:], y[TRAIN_TEST_split:]


class LinearRegressionModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.LINEAR_LAYER = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.LINEAR_LAYER(x)

def plot_Predictions(Predictions=None) -> None:
    plt.scatter(X_train, y_train, label="Training Data")
    plt.scatter(X_test, y_test, label="Test Data")

    if Predictions is not None:
        plt.scatter(X_test, Predictions)

    plt.legend()
    plt.show()

def save_model(model: LinearRegressionModel) -> None:
    MODEL_PATH: Path = Path("model")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME: str = "RNN_MODEL.pth"
    MODEL_SAVE_PATH: Path = MODEL_PATH / MODEL_NAME

    torch.save(obj=model.state_dict(), f=MODEL_SAVE_PATH)

def load_model() -> Dict[str, Any]:
    MODEL_SAVE_PATH: str = "model/RNN_MODEL.pth"

    loaded_model: LinearRegressionModel = LinearRegressionModel()
    loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    print(loaded_model.state_dict())

    return loaded_model.state_dict()

def main() -> None:
    torch.manual_seed(42)
    model: LinearRegressionModel = LinearRegressionModel()

    # with torch.inference_mode():
    #     y_predictions = model(X_test)
    # plot_Predictions(y_predictions)

    LOSS_FN: nn.L1Loss = nn.L1Loss()
    OPTIMIZER: torch.optim.SGD = torch.optim.SGD(params=model.parameters(), lr=0.01)
    
    epochs: int = 200
    
    train_loss: list[int] = []
    tests_loss: list = []
    epochs_count: list = []

    for epoch in range(epochs):
        model.train()

        y_predictions: Any = model(X_train)
        loss: Any = LOSS_FN(y_predictions, y_train)
        
        OPTIMIZER.zero_grad()
        loss.backward()

        OPTIMIZER.step()

        model.eval()
        
        with torch.inference_mode():
            test_pred: Any = model(X_test)
            test_loss: Any = LOSS_FN(test_pred, y_test.type(torch.float))
            
        if epoch % 10 == 0:
            epochs_count.append(epoch)
            train_loss.append(loss.detach().numpy())
            tests_loss.append(test_loss.detach().numpy())

            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")
    
    with torch.inference_mode():
        y_predictions_new: Any = model(X_test)
        
        print(X_train)
        print(y_predictions_new)

        plot_Predictions(Predictions= y_predictions_new)

    plt.plot(epochs_count, train_loss, label="Train loss")
    plt.plot(epochs_count, tests_loss, label="Test Loss")
    plt.title("Training and test loss curves")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    plt.show()

    # save_model(model)

if __name__ == "__main__":
    main()
    # load_model()

