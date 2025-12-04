import sys

from preprocessing import load_data, split_data, prepare_features
from model_training import train_model, save_model, load_model
from metrics import evaluate, mape
from plots import (
    plot_corr,
    plot_feature_importance,
    plot_distributions,
    plot_boxplots,
    plot_errors,
    plot_scatter,
)


def main_menu():
    print("1. Обучить модель")
    print("2. Оценить модель")
    print("3. Построить корреляцию")
    print("4. Важность признаков")
    print("5. Распределения признаков")
    print("6. Boxplot признаков")
    print("7. Ошибки модели")
    print("8. Scatter по топ-признакам")
    print("9. Выход")


def main():
    while True:
        main_menu()
        choice = input("Введите номер: ").strip()

        if choice == "1":
            df = load_data()
            X_train, X_test, y_train, y_test = split_data(df)
            model = train_model(X_train, y_train, X_test, y_test)
            save_model(model)
            print("Модель обучена и сохранена.")
        elif choice == "2":
            df = load_data()
            _, X_test, _, y_test = split_data(df)
            model = load_model()
            eval_res = evaluate(model, X_test, y_test)
            mape_val = mape(model, X_test, y_test)
            print("MAPE:", mape_val)
        elif choice == "3":
            df = load_data()
            plot_corr(df)
            print("График корреляции сохранён.")
        elif choice == "4":
            df = load_data()
            X, y = prepare_features(df)
            tmp = X.copy()
            tmp["rank"] = y
            model = load_model()
            plot_feature_importance(model, tmp)
            print("График важности признаков сохранён.")
        elif choice == "5":
            df = load_data()
            plot_distributions(df)
            print("Графики распределений сохранены.")
        elif choice == "6":
            df = load_data()
            plot_boxplots(df)
            print("Boxplot-графики сохранены.")
        elif choice == "7":
            df = load_data()
            _, X_test, _, y_test = split_data(df)
            model = load_model()
            plot_errors(model, X_test, y_test)
            print("График ошибок сохранён.")
        elif choice == "8":
            df = load_data()
            X, y = prepare_features(df)
            tmp = X.copy()
            tmp["rank"] = y
            plot_scatter(tmp)
            print("Scatter-графики сохранены.")
        elif choice == "9":
            sys.exit(0)
        else:
            print("Неверный ввод.")


if __name__ == "__main__":
    main()
