program mnist_mlp;

{$H+}

uses
  SysUtils,
  noe,
  noe.Math,
  noe.neuralnet,
  noe.optimizer,
  noe.utils;

var
  DatasetTrain, DatasetTest: TTensor;
  Xtrain, ytrain, Xtest, ytest: TTensor;
  Loss, yPred, yPredTest, Xbatch, ybatch, v: TVariable;
  encoder: TOneHotEncoder;
  br: TBatchingResult;
  NNModel: TModel;
  optim: TAdamOptimizer;
  BatchLoss, ValidationAcc: double;
  i, j: integer;

begin
  randomize;
  WriteLn('Loading datasets...');

  { Joseph Redmond's CSV version, https://pjreddie.com/projects/mnist-in-csv/.
    They are not included in noe repository due to the size. }
  DatasetTrain := ReadCSV('mnist_train.csv');
  DatasetTest  := ReadCSV('mnist_test.csv');

  Xtrain := GetColumnRange(DatasetTrain, 1, 28 * 28) / 255;
  ytrain := GetColumn(DatasetTrain, 0);
  Xtest  := GetColumnRange(DatasetTest, 1, 28 * 28) / 255;
  ytest  := GetColumn(DatasetTest, 0);

  encoder := TOneHotEncoder.Create;
  ytrain  := encoder.Encode(ytrain);

  NNModel := TModel.Create([
    TDenseLayer.Create(28 * 28, 512),
    TReLULayer.Create,
    TDropoutLayer.Create(0.2),
    TDenseLayer.Create(512, 512),
    TReLULayer.Create,
    TDropoutLayer.Create(0.2),
    TDenseLayer.Create(512, 10),
    TSoftMaxLayer.Create(1)
    ]);

  br := CreateBatch(Xtrain, ytrain, 1000);
  Xtrain.Free;

  optim := TAdamOptimizer.Create;
  optim.LearningRate := 0.0001;
  optim.Verbose := False;

  WriteLn('Start training.');
  for i := 1 to 50 do
  begin
    BatchLoss := 0;
    for j := 0 to br.BatchCount - 1 do
    begin
      Xbatch := br.Xbatches[j];
      ybatch := br.ybatches[j];

      yPred := NNModel.Eval(Xbatch);
      Loss  := CrossEntropyLoss(yPred, ybatch) + L2Regularization(NNModel);

      optim.UpdateParams(Loss, NNModel.Params);

      BatchLoss := BatchLoss + Loss.Data.Val[0];
    end;

    GLOBAL_SKIP_GRAD := True;
    yPredTest     := NNModel.Eval(Xtest);
    ValidationAcc := AccuracyScore(encoder.Decode(yPredTest.Data), ytest);
    GLOBAL_SKIP_GRAD := False;
    WriteLn(Format('Epoch %d: batch_mean_loss=%f; validation_acc=%f',
      [i, BatchLoss / br.BatchCount, ValidationAcc]));
  end;

  SaveModel(NNModel, 'mnist_mlp.model');
  WriteLn('Model was saved.');

  ReadLn;
end.
