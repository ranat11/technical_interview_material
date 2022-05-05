import argparse
import regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader

import pickle
import matplotlib.pyplot as plt


def train(render=True, filename="model"):
    # load the dataset
    train_dataset = regression.BicepCDataset(
        csv_file="data/regression_train.csv", transform=False)

    print(f"there are {len(train_dataset)} instances in the dataset")

    train_dataloader = DataLoader(
        dataset=train_dataset, batch_size=len(train_dataset), shuffle=True)
    features, targets = next(iter(train_dataloader))

    # train data
    model = LinearRegression(fit_intercept=True, copy_X=True, n_jobs=None)
    model.fit(features, targets)
    print(f"Parameters M: {model.coef_}")
    print(f"Parameters b: {model.intercept_}")

    print("train model ended")

    # evaluation
    pred = model.predict(features)
    print(
        f"train: Mean Squared Error: {mean_squared_error(targets, pred):.2f}")
    if render:
        fig, ax = plt.subplots(figsize=(8, 8))
        regression.plot_regression_results(
            ax, targets, pred, 'LinearRegression', f'MSE={mean_squared_error(targets, pred):.2f} cm', "BicepC")
        plt.tight_layout()
        plt.show()

    # save model
    savefile = f'model/{filename}.sav'
    pickle.dump(model, open(savefile, 'wb'))


def test(render=True, filename="model"):
    # load model
    loadfile = f'model/{filename}.sav'
    model = pickle.load(open(loadfile, 'rb'))

    # test the model
    test_dataset = regression.BicepCDataset(
        csv_file="data/regression_test.csv", transform=False)

    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=True)
    features, targets = next(iter(test_dataloader))
    pred = model.predict(features)
    print(
        f"test: Mean Squared Error: {mean_squared_error(targets, pred):.2f}")

    if render:
        fig, ax = plt.subplots(figsize=(8, 8))
        regression.plot_regression_results(
            ax, targets, pred, 'LinearRegression test', f'MSE={mean_squared_error(targets, pred):.2f} cm', "BicepC")
        plt.tight_layout()
        plt.show()


def main(args):
    if args.train:
        train(render=args.render, filename=args.file_name)

    test(render=args.render, filename=args.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train", help="train model with linear regression", action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--file-name", help="file name for saving and loading", default="model", type=str)
    parser.add_argument(
        "--render", help="plot graph between prediction and real target", action=argparse.BooleanOptionalAction)

    args = parser.parse_args()
    main(args)
