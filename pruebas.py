import matplotlib.pyplot as plt
import numpy as np

def pruebas():
    # Datos
    train_cnn = [668, 515, 1704]
    test_cnn = [8, 658, 51]
    val_cnn = [0, 711, 0]

    train_mlp = [413, 703, 1516]
    test_mlp = [0, 444, 265]
    val_mlp = [0, 445, 266]

    # Categorías
    labels = ["Buy", "Sell", "Hold"]

    # --- CNN ---
    plt.figure(figsize=(7, 4))
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, train_cnn, width, color="skyblue", label="Train")
    plt.bar(x, test_cnn, width, color="cornflowerblue", label="Test")
    plt.bar(x + width, val_cnn, width, color="navy", label="Validation")

    plt.title("CNN — Class Balance")
    plt.xlabel("Signals Type")
    plt.ylabel("Amount of Signals")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # --- MLP ---
    plt.figure(figsize=(7, 4))
    plt.bar(x - width, train_mlp, width, color="skyblue", label="Train")
    plt.bar(x, test_mlp, width, color="cornflowerblue", label="Test")
    plt.bar(x + width, val_mlp, width, color="navy", label="Validation")

    plt.title("MLP — Class Balance")
    plt.xlabel("Signals Type")
    plt.ylabel("Amount of Signals")
    plt.xticks(x, labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    pruebas()


    