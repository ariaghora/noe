{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 - OBJECTIVE
   =========
   This program demonstrates the use of neural network TModel class, TLayer
   class and its derivatives. Instead of defining model weights manually, the
   TLayer provides a wrapper to avoid doing so. The problem is optical digit
   classification.

 - DATASET DESCRIPTION
   ===================
   From "archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits":
   "We used preprocessing programs made available by NIST to extract normalized
   bitmaps of handwritten digits from a preprinted form. From a total of 43
   people, 30 contributed to the training set and different 13 to the test set.
   32x32 bitmaps are divided into nonoverlapping blocks of 4x4 and the number of
   on pixels are counted in each block. This generates an input matrix of 8x8
   where each element is an integer in the range 0..16. This reduces dimensionality
   and gives invariance to small distortions."
}

program optdigits_hl_api;

{$mode objfpc}{$H+}

uses
  math,
  noe,
  noe.Math,
  noe.optimizer,
  noe.utils,
  noe.neuralnet;

const
  MAX_EPOCH = 100;

var
  i, M, NInputNeuron, NOutputNeuron, PredictedLabel, ActualLabel, SampleIdx: longint;
  DatasetTrain, DatasetTest, LabelsTest, FeatsTest, ImageSample: TTensor;
  Xtrain, ytrain, ypred, Loss, ypredTest: TVariable;
  LabelEncoder: TOneHotEncoder;
  NNModel: TModel;
  optimizer: TAdamOptimizer;
  TrainingAcc, TestingAcc: double;

  foo : TTensor;

begin
  RandSeed := 1;

  { Data preparation ----------------------------------------------------------}
  WriteLn('Loading and preparing the data...');
  DatasetTrain := ReadCSV('../datasets/optdigits-train.csv');
  M := DatasetTrain.Shape[0];

  Xtrain := GetRange(DatasetTrain, 0, 0, M, 64);
  ytrain := Squeeze(GetColumn(DatasetTrain, 64));
  LabelEncoder := TOneHotEncoder.Create;
  ytrain := LabelEncoder.Encode(ytrain.Data);

  { Model preparation ---------------------------------------------------------}
  NInputNeuron  := Xtrain.Shape[1];
  NOutputNeuron := ytrain.Shape[1];

  NNModel := TModel.Create([
    TDenseLayer.Create(NInputNeuron, 128),
    TLeakyReLULayer.Create(0.3),
    TDenseLayer.Create(128, 64),
    TLeakyReLULayer.Create(0.3),
    TDenseLayer.Create(64, NOutputNeuron),
    TSoftMaxLayer.Create(1)
  ]);

  { Training phase ------------------------------------------------------------}
  WriteLn('Press enter to start training in ', MAX_EPOCH, ' iterations.');
  ReadLn;
  optimizer := TAdamOptimizer.Create;
  for i := 0 to MAX_EPOCH - 1 do
  begin
    ypred := NNModel.Eval(Xtrain);
    Loss  := CrossEntropyLoss(ypred, ytrain) + L2Regularization(NNModel);

    optimizer.UpdateParams(Loss, NNModel.Params);
  end;

  TrainingAcc := AccuracyScore(LabelEncoder.Decode(ypred.Data),
    LabelEncoder.Decode(ytrain.Data));
  WriteLn('Training accuracy: ', TrainingAcc: 2: 4);
  WriteLn;


  { Testing Phase -------------------------------------------------------------}
  WriteLn('Traning completed. Now evaluating the model on the testing set...');
  DatasetTest := ReadCSV('../datasets/optdigits-test.csv');
  FeatsTest   := GetRange(DatasetTest, 0, 0, DatasetTest.Shape[0], 64) / 16;
  LabelsTest  := Squeeze(GetColumn(DatasetTest, 64));

  ypredTest  := NNModel.Eval(FeatsTest.ToVariable());
  TestingAcc := AccuracyScore(LabelEncoder.Decode(ypredTest.Data),
    LabelsTest);
  WriteLn('testing accuracy = ', TestingAcc: 2: 2);

  { Pick one sample from the test set. Let's try to visualize and predict the
    label }
  SampleIdx   := 100;
  ImageSample := GetRow(FeatsTest, SampleIdx, True);
  ypredTest   := NNModel.Eval(ImageSample.ToVariable(False));

  { transform the probability into the discrete label }
  PredictedLabel := Round(LabelEncoder.Decode(ypredTest.Data).Val[0]);
  ActualLabel    := Round(LabelsTest.GetAt(SampleIdx));

  WriteLn('Predicting one of the test samples...');
  VisualizeMatrix(ImageSample.Reshape([8, 8]));
  WriteLn('Predicted class: ', PredictedLabel, '; Probability: ', Max(ypredTest.Data,
    1).Val[0]: 2: 2, '; The actual class: ', ActualLabel);

  ReadLn;
end.
