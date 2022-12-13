program iris;

{$mode objfpc}{$H+}

uses
  SysUtils, DateUtils, multiarray, numerik,
  noe, noe.optimizer, noe.neuralnet;

var
  Dataset, X, Y, YBin, YPred, Loss: TTensor;
  model: TNNModel;
  opt: TOptAdam;
  i: integer;
  t: TDateTime;

begin
  Dataset := ReadCSV('../datasets/iris.csv');

  X := Dataset[[ _ALL_, Range(0, 4) ]]; // Get all rows and first four columns
  Y := Dataset[[ _ALL_, 4 ]]; // Get all rows and a column with index 4
  YBin := BinarizeLabel(Y); // Transform labels into one-hot vectors

  model := TNNModel.Create;
  model.AddLayer(TLayerDense.Create(4, 30));
  model.AddLayer(TLayerReLU.Create());
  model.AddLayer(TLayerDense.Create(30, 3));
  model.AddLayer(TLayerSoftmax.Create(1));

  opt := TOptAdam.Create(model.Params); // Adam optimizer
  opt.LearningRate := 0.01;

  t := Now;
  for i := 0 to 100 do
  begin
    YPred := model.Eval(X);
    Loss := CrossEntropy(YPred, YBin);
    Loss.Backward();
    opt.Step;

    if i mod 10 = 0 then
      WriteLn('Loss at iteration ', i, ': ', Loss.Data.Get(0) : 5 : 2);
  end;

  WriteLn('Training completed in ', MilliSecondsBetween(Now, t), ' ms');
  WriteLn('Training accuracy: ', Mean(ArgMax(YPred.Data, 1, True)).Item : 5 : 2);
  WriteLn('Press enter to exit'); ReadLn;

  model.Free;
  opt.Free;
end.

