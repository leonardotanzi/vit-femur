

if __name__=="__main__":
    model = load_model("model.h5")
    predicted_classes = np.argmax(model.predict(valid_gen, steps=valid_gen.n // valid_gen.batch_size + 1), axis=1)
    true_classes = valid_gen.classes
    class_labels = list(valid_gen.class_indices.keys())

    confusionmatrix = confusion_matrix(true_classes, predicted_classes)
    print(confusionmatrix)
    plt.figure(figsize=(16, 16))
    sns.heatmap(confusionmatrix, cmap='Blues', annot=True, cbar=True)

    print(classification_report(true_classes, predicted_classes))