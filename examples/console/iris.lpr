program iris;

{$mode objfpc}{$H+}

uses
  noe,
  noe.Math,
  noe.neuralnet,
  noe.utils,
  noe.optimizer;

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

  X := GetColumnRange(Dataset, 0, 4);
  X := StandardScaler(X);
  y := GetColumn(Dataset, 4);

  { One-hot encoding the raw label }
  Enc := TOneHotEncoder.Create;
  y   := Enc.Encode(y);

  MyModel := TModel.Create([
    TDenseLayer.Create(4, 30),
    TLeakyReLULayer.Create(0.1),
    TDenseLayer.Create(30, 3),
    TSoftMaxLayer.Create(1)
  ]);

  opt := TAdamOptimizer.Create;

  opt.LearningRate := 0.01;

  { To show loss value at each iteration. Default: True }
  opt.Verbose := True;

  for i := 0 to 1 do
  begin
    yPred := MyModel.Eval(X);
    Loss  := CrossEntropyLoss(yPred, y);
    opt.UpdateParams(Loss, MyModel.Params);
  end;

  WriteLn('Tranining accuracy: ', AccuracyScore(Enc.Decode(yPred.Data),
    Enc.Decode(y)): 2: 3);

  noe.Cleanup;
  Enc.Cleanup;
  MyModel.Cleanup;

  ReadLn;
end.
