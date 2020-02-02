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
  noe,
  noe.Math,
  noe.neuralnet,
  noe.optimizer,
  noe.utils;

const
  MAX_EPOCH = 50;

var
  i, M, NInputNeuron, NOutputNeuron, PredictedLabel, ActualLabel,
  NHiddenNeuron, SampleIdx: longint;
  DatasetTrain, FeatsTrain, LabelsTrain, EncodedLabelsTrain, DatasetTest,
  LabelsTest, FeatsTest, ImageSample: TTensor;
  Xtrain, ytrain, ypred, Loss, ypredTest: TVariable;
  LabelEncoder: TOneHotEncoder;
  NNModel: TModel;
  optimizer: TAdamOptimizer;
  TrainingAcc, TestingAcc: double;

begin
  RandSeed := 1;

  { Data preparation ----------------------------------------------------------}
  DatasetTrain := ReadCSV('../datasets/optdigits-train.csv');
  M := DatasetTrain.Shape[0];

  FeatsTrain   := GetRange(DatasetTrain, 0, 0, M, 64) / 16;
  LabelsTrain  := Squeeze(GetColumn(DatasetTrain, 64));
  LabelEncoder := TOneHotEncoder.Create;
  EncodedLabelsTrain := LabelEncoder.Encode(LabelsTrain);

  Xtrain := TVariable.Create(FeatsTrain);
  ytrain := TVariable.Create(EncodedLabelsTrain);

  { Model preparation ---------------------------------------------------------}
  NInputNeuron  := Xtrain.Shape[1];
  NHiddenNeuron := 32;
  NOutputNeuron := ytrain.Shape[1];

  NNModel := TModel.Create([
    TDenseLayer.Create(NInputNeuron, NHiddenNeuron, atReLU),
    TDenseLayer.Create(NHiddenNeuron, NOutputNeuron, atNone),
    TSoftMaxLayer.Create(1)
    ]);

  { Training phase ------------------------------------------------------------}
  optimizer := TAdamOptimizer.Create;
  optimizer.LearningRate := 0.003;
  for i := 0 to MAX_EPOCH - 1 do
  begin
    ypred := NNModel.Eval(Xtrain);
    Loss  := CrossEntropyLoss(ypred, ytrain) + L2Regularization(NNModel);

    optimizer.UpdateParams(Loss, NNModel.Params);

    TrainingAcc := AccuracyScore(LabelEncoder.Decode(ypred.Data), LabelsTrain);
    WriteLn('Training accuracy: ', TrainingAcc: 2: 4);
  end;

  { Testing Phase -------------------------------------------------------------}

  WriteLn('Traning completed. Now evaluating the model on the testing set...');

  DatasetTest := ReadCSV('../datasets/optdigits-test.csv');
  FeatsTest   := GetRange(DatasetTest, 0, 0, DatasetTest.Shape[0], 64) / 16;
  LabelsTest  := Squeeze(GetColumn(DatasetTest, 64));

  ypredTest  := NNModel.Eval(FeatsTest.ToVariable());
  TestingAcc := AccuracyScore(LabelEncoder.Decode(ypredTest.Data), LabelsTest);
  WriteLn('testing accuracy = ', TestingAcc: 2: 2);

  { Pick one sample from the test set. Let's try to visualize and predict the
    label }
  SampleIdx   := 300;
  ImageSample := GetRow(FeatsTest, SampleIdx);
  ypredTest   := NNModel.Eval(ImageSample.ToVariable(False));

  { transform the probability into the discrete label }
  PredictedLabel := Round(LabelEncoder.Decode(ypredTest.Data).Val[0]);
  ActualLabel    := Round(LabelsTest.GetAt(SampleIdx));
  WriteLn('Predicted: ', PredictedLabel, '; Prob: ', Max(ypredTest.Data,
    1).Val[0]: 2: 2, '; Actual: ', ActualLabel);

  ReadLn;
end.
