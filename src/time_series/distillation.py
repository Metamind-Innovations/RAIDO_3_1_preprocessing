import torch
import pandas as pd


def prepare_data(csv_file):
    df = pd.read_csv(csv_file, sep=';', parse_dates=[0], dayfirst=True, low_memory=False)
    df['timestamp'] = df[df.columns[0]].astype(int) / 10 ** 9
    prepared_data = df[['timestamp', df.columns[1]]].values.astype(float)
    return prepared_data[:, 0].reshape(-1, 1), prepared_data[:, 1]


# RCIG distillation implementation on prepared data.
def rcig_distillation(X, y, distilled_size=100, steps=100):
    input_dim = X.shape[1]
    unique_classes = torch.unique(torch.tensor(y))
    output_dim = len(unique_classes)

    # Map original target values to class indices
    class_to_index = {cls.item(): idx for idx, cls in enumerate(unique_classes)}
    y_mapped = torch.tensor([class_to_index[val] for val in y], dtype=torch.long)

    # Initialize synthetic dataset.
    distilled_X = torch.randn(distilled_size, input_dim, requires_grad=True)
    distilled_y = y_mapped[:distilled_size]  # Ensure using mapped target values

    model = torch.nn.Linear(input_dim, output_dim)
    optimizer = torch.optim.Adam([distilled_X], lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()  # Use CrossEntropyLoss for classification

    for step in range(steps):
        optimizer.zero_grad()
        outputs = model(distilled_X)
        loss = criterion(outputs, distilled_y)  # No need to squeeze or change dtype
        loss.backward()
        optimizer.step()

    return distilled_X.detach().numpy(), distilled_y.detach().numpy()


X, y = prepare_data('power_small.csv')
distilled_X_kip, distilled_y_kip = rcig_distillation(torch.tensor(X, dtype=torch.float32), y)

print(f'Distilled Data (KIP): {distilled_X_kip}, {distilled_y_kip}')
