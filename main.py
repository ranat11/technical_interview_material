import regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt


def main():
    # load the dataset
    train_dataset = regression.BicepCDataset(
        csv_file="data/regression_train.csv", transform=False)

    print(f"there are {len(train_dataset)} instances in the dataset")

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    train_features, train_targets = next(iter(train_dataloader))

    # train data
    model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
    model.fit(train_features, train_targets)
    print(f"Parameters M: {model.coef_}")
    print(f"Parameters b: {model.intercept_}")

    print("train model ended")

    # evaluation
    train_pred = model.predict(train_features)
    print(
        f"Mean Squared Error: {mean_squared_error(train_targets, train_pred):.2f}")
    fig, ax = plt.subplots(figsize=(8, 8))
    regression.plot_regression_results(
        ax, train_targets, train_pred, 'LinearRegression', f'MSE={mean_squared_error(train_targets, train_pred):.2f} cm', "BicepC")

    plt.tight_layout()
    plt.show()

    # test the model
    test_dataset = regression.BicepCDataset(
        csv_file="data/regression_test.csv", transform=False)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
    test_features, test_targets = next(iter(test_dataloader))
    test_pred = model.predict(test_features)

    fig, ax = plt.subplots(figsize=(8, 8))
    regression.plot_regression_results(
        ax, test_targets, test_pred, 'LinearRegression test', f'MSE={mean_squared_error(test_targets, test_pred):.2f} cm', "BicepC")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
