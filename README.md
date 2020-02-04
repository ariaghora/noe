<div align="center">
<img src="assets/noe-txt.png" alt="logo" width="200px"></img>
</div>

***

<div align="center">
  
[![Generic badge](https://img.shields.io/badge/license-MIT-blue.svg)](https://shields.io/)
[![made-for-pascal](https://img.shields.io/badge/Made%20for-object%20pascal-e9448a.svg)](https://code.visualstudio.com/)

</div>

Noe is a framework for an easier scientific computation in object pascal, especially to build neural networks, and hence the name — *noe (Korean:뇌) means brain*. It supports the creation of arbitrary rank tensor and its arithmetical operations. Some of the key features:
- Automatic gradient computation
- Numpy-style broadcasting
- Interface with *OpenBLAS* for some heavy-lifting
- Interface with *GNU plot* for plotting

Noe also provides several tensor creation and preprocessing helper functionalities. The created tensors are then wrapped inside the TVariable to train the neural network.

```delphi
{ Load and prepare the data. }
Dataset := ReadCSV('iris.csv');
X       := GetColumnRange(Dataset, 0, 4);
X       := StandardScaler(X);

Enc := TOneHotEncoder.Create;    
y   := GetColumn(Dataset, 4);
y   := Enc.Encode(Squeeze(y));

{ Wrap tensor in a TVariable }
XVar := X.ToVariable();
yVar := y.ToVariable();
```

With autograd, it is possible to make of neural networks in various degree of abstraction. You can control the flow of of the network, even design a custom fancy loss function. For the high level API, there are several implementation of neural network layers, optimzier, along with TModel class helper, so you can prototype your network quickly.
```delphi
NNModel := TModel.Create([
  TDenseLayer.Create(NInputNeuron, 32, atReLU),
  TDropoutLayer.Create(0.2),
  TDenseLayer.Create(32, 16, atNone),
  TDropoutLayer.Create(0.2),
  TDenseLayer.Create(16, NOutputNeuron, atNone),
  TSoftMaxLayer.Create(1)
]);

optimizer := TAdamOptimizer.Create;
optimizer.LearningRate := 0.003;
for i := 0 to MAX_EPOCH - 1 do
begin
  { Make a prediction and compute the loss }
  yPred := NNModel.Eval(XVar);
  Loss  := CrossEntropyLoss(yPred, yVar) + L2Regularization(NNModel);
  
  { Update model parameter w.r.t. the loss }
  optimizer.UpdateParams(Loss, NNModel.Params);
end;
```

Check out [the wiki](https://github.com/ariaghora/noe/wiki) for more documentation. Please note that this framework is developed and heavily tested using fpc 3.0.4, with object pascal syntax mode, on a windows machine. Portability is not really my first concern right now, but any helps are sincerely welcome.

> **A blatant disclaimer -** *This is my learning and experimental project. The development is still early and active. That said, the use for production is not encouraged at this moment.*
