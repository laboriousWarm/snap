import torch
import torch.nn.functional as nn
import numpy as np
import  pandas as pd

names = [
        "age",
        "workclass",
        "fnlwgt",
        "education",
        "education-num",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
        "native-country",
        "class",
    ]
cat_cols = [
        "workclass",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

cont_cols = [
        "age",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]

df = pd.read_csv('dataset/adult.data', names=names)
df = df.iloc[:1000, :]

df['class'] = df['class'].astype('category').cat.codes
df[cat_cols] = df[cat_cols].astype('category')
# print(df)

drop = ["fnlwgt",
        "education"]
df = df.drop(drop, axis=1)
# print(df.dtypes)

df_one_hot = pd.get_dummies(df, columns=cat_cols)
df_one_hot['class'] = df_one_hot.pop('class')
df_one_hot.replace({True: 1, False: 0}, inplace=True)
print(df_one_hot)

df_one_hot[cont_cols] = df_one_hot[cont_cols] / (df_one_hot[cont_cols].max())
# 近似为3位小数
df_one_hot[cont_cols] = df_one_hot[cont_cols].round(3)
print(df_one_hot.dtypes)


labels = torch.Tensor(df_one_hot['class'].values)
print(labels.shape)
print(labels.squeeze().shape)

del df_one_hot['class']

data = torch.Tensor(df_one_hot.values)

dataset = torch.utils.data.TensorDataset(
        data, labels.squeeze().to(dtype=torch.long)
)

# print()

dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=8,
        shuffle=True
)

print(dataloader)



class NeuralNet(torch.nn.Module):
    """PyTorch implementation of a multilayer perceptron with ReLU activations"""

    def __init__(
        self,
        input_dim: int,
        layer_sizes:torch.List[int] = [32, 16],
        num_classes: int = 2,
        dropout: bool = False,
    ):
        super(NeuralNet, self).__init__()
        self._input_dim = input_dim
        self._layer_sizes = layer_sizes
        self._num_classes = num_classes
        # print(self._layer_sizes)
        layers = [torch.nn.Linear(input_dim, layer_sizes[0])]
        layers.append(torch.nn.ReLU())
        if dropout:
            layers.append(torch.nn.Dropout())

        # Initialize all layers according to sizes in list
        for i in range(len(self._layer_sizes) - 1):
            layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            layers.append(torch.nn.ReLU())
            if dropout:
                layers.append(torch.nn.Dropout())
        layers.append(torch.nn.Linear(layer_sizes[-1], num_classes))

        # Wrap layers in ModuleList so PyTorch
        # can compute gradients
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

model = NeuralNet(
        input_dim=76,
        num_classes=2
)
model = model.to('cpu')
optimizer = torch.optim.Adam(model.parameters(), **{"lr": 0.03, "weight_decay": 0.0001})
criterion = torch.nn.CrossEntropyLoss()

for (data, label) in dataloader:
    data = data.to('cpu')
    # print(data.shape)
    label = label.to('cpu')
    print(label.shape)
    optimizer.zero_grad()
    out = model.forward(data)
    print(out.shape)
    loss = criterion(out, label)
    loss.backward()
    optimizer.step()