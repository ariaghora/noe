program iris;

{$mode objfpc}{$H+}

uses
  noe,
  noe.Math,
  noe.neuralnet,
  noe.utils,
  noe.optimizer,
  noe.backend.blas;

var
  X, y: TTensor;
  yPred, Loss: TVariable;
  Enc: TOneHotEncoder;
  Dataset: TTensor;
  MyModel: TModel;
  opt: TAdamOptimizer;
  i: longint;

begin
  Dataset := ReadCSV('../datasets/iris.csv');
  Enc     := TOneHotEncoder.Create; // One-hot encoding

  X := GetColumnRange(Dataset, 0, 4);
  X := StandardScaler(X);
  y := GetColumn(Dataset, 4);
  y := Enc.Encode(y);

  MyModel := TModel.Create([
    TDenseLayer.Create(4, 30),
    TLeakyReLULayer.Create(0.1),
    TDenseLayer.Create(30, 3),
    TSoftMaxLayer.Create(1)
  ]);

  opt := TAdamOptimizer.Create;

  opt.LearningRate := 0.01;
  opt.Verbose      := True; // To show loss value at each iteration
                            // Default: True

  for i := 0 to 100 do
  begin
    yPred := MyModel.Eval(X);
    Loss  := CrossEntropyLoss(yPred, y);
    opt.UpdateParams(Loss, MyModel.Params);
  end;

  WriteLn('Tranining accuracy: ', AccuracyScore(Enc.Decode(yPred.Data),
    Enc.Decode(y)): 2: 3);

  ReadLn;
end.
