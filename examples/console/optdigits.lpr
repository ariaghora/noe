{
 This file is part of "noe" library.

 Noe library. Copyright (C) 2020 Aria Ghora Prabono.

 - OBJECTIVE
   =========
   This program highlights several important high (abstraction) level features
   of noe through the case of optical digits classification problem. The input
   is handwritten digits datasaet as described below.

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
program optdigits;

{$mode objfpc}{$H+}

uses
  SysUtils,
  Math,
  noe,
  noe.Math,
  noe.utils,
  noe.optimizer,
  noe.plot.gnuplot;

const
  MAX_EPOCH = 150;

var
  DatasetTrain, DatasetTest, FeatsTrain, LabelsTrain, EncodedLabelsTrain,
  Losses, ImageSample: TTensor;
  Lambda, TrainingAcc, TestingAcc: double;
  FeatsTest, LabelsTest, ypredTest: TTensor;
  i, M, NHiddenNeuron, NInputNeuron, NOutputNeuron, SampleIdx,
  PredictedLabel, ActualLabel: longint;
  Xtrain, ytrain, ypred, W1, W2, b1, b2, L2Reg, CrossEntropyLoss, TotalLoss: TVariable;

  Optimizer: TAdamOptimizer;
  LabelEncoder: TOneHotEncoder;

  { A tiny function to obtain classification accuracy. It simply computes the
    number of correctly classified samples divided by the total number of
    samples. }
  function AccuracyScore(predicted, actual: TTensor): double;
  var
    i: integer;
    tot: double;
  begin
    tot := 0;
    for i := 0 to predicted.Size - 1 do
      { check if the sample is correctly classified (i.e., predicted = actual) }
      if predicted.GetAt(i) = actual.GetAt(i) then
        tot := tot + 1;
    Result  := tot / predicted.Size;
  end;

  { A procedure to display figure using noe's interface to GNU plot. }
  procedure ShowFigure(Losses: TTensor; Title: string; PlotType: TPlotType);
  var
    Figure: TFigure;
    Plot: TPlot;
  begin
    GNUPlotInit('gnuplot');
    Figure := TFigure.Create;
    Figure.Title := Title;

    if PlotType = ptImage then
      Figure.Palette := 'gray';

    Plot := TPlot.Create;
    Plot.Title := Title;
    Plot.PlotType := PlotType;
    Plot.SetDataPoints(Losses);

    Figure.AddPlot(Plot);
    Figure.Show;
  end;

begin
  RandSeed := 1;

  { Load the DatasetTrain from CSV. Noe has a built-in function to do so. }
  DatasetTrain := ReadCSV('../datasets/optdigits-train.csv');

  M := DatasetTrain.Shape[0]; // The number of samples

  { Get the columns that represent feature vectors. The opdigits DatasetTrain contains
    values within the range [0, 16]. Thus we can perform feature scaling by simply
    dividing the feature values by 16. }
  FeatsTrain := GetRange(DatasetTrain, 0, 0, M, 64) / 16;

  { The column containing LabelsTrain is located at index 64. }
  LabelsTrain := Squeeze(GetColumn(DatasetTrain, 64));

  { Convert the categorical label into one-hot encoded matrix. }
  LabelEncoder := TOneHotEncoder.Create;
  EncodedLabelsTrain := LabelEncoder.Encode(LabelsTrain);

  { Then we use TVariable to wrap around the features and LabelsTrain. }
  Xtrain := TVariable.Create(FeatsTrain);
  ytrain := TVariable.Create(EncodedLabelsTrain);

  NInputNeuron  := Xtrain.Shape[1]; // The number of features (columns)
  NHiddenNeuron := 32; // Feel free to experiment with the value.
  NOutputNeuron := ytrain.Shape[1]; // The number of unique class in the LabelsTrain

  { Initialize weights and biases. The weights are randomized, and the biases
    are set to a particular value. Typically the value is small in the beginning.
    Some implementations just use 1/sqrt(n_of_layer_neuron) for the initial bias
    value. }
  W1 := RandomTensorNormal([NInputNeuron, NHiddenNeuron]);
  W2 := RandomTensorNormal([NHiddenNeuron, NOutputNeuron]);
  b1 := CreateTensor([1, NHiddenNeuron], 1 / NHiddenNeuron ** 0.5);
  b2 := CreateTensor([1, NOutputNeuron], 1 / NOutputNeuron ** 0.5);

  { Since we need the gradient of weights and biases, it is mandatory to set
    RequiresGrad property to True. We can also set the parameter individually
    for each parameter, e.g., `W1.RequiresGrad := True;`. }
  SetRequiresGrad([W1, W2, b1, b2], True);

  { Noe provides the implementation of several optimization algorithms. For this
    example we will use adam optimizer. }
  Optimizer := TAdamOptimizer.Create;

  { The default is 0.001. Feel free to experiment with the value. }
  Optimizer.LearningRate := 0.01;

  Lambda := 0.001; // Weight decay. Feel free to experiment with the value.

  { Keep track the loss values over iteration }
  Losses := CreateEmptyTensor([MAX_EPOCH]);

  for i := 0 to MAX_EPOCH - 1 do
  begin
    { Our neural network -> ŷ = softmax(σ(XW₁ + b₁)W₂ + b₂). }
    ypred := SoftMax(ReLU(Xtrain.Dot(W1) + b1).Dot(W2) + b2, 1);

    { Compute the cross-entropy loss. }
    CrossEntropyLoss := -Sum(ytrain * Log(ypred)) / M;

    { Compute L2 regularization term. Later it is added to the total loss to
      prevent model overfitting. }
    L2Reg := Sum(W1 * W1) + Sum(W2 * W2);

    TotalLoss := CrossEntropyLoss + (Lambda / (2 * M)) * L2Reg;
    Losses.SetAt(i, TotalLoss.Data.GetAt(0));

    { Update the network weight }
    Optimizer.UpdateParams(TotalLoss, [W1, W2, b1, b2]);

    TrainingAcc := AccuracyScore(LabelEncoder.Decode(ypred.Data), LabelsTrain);
    Writeln('Epoch ', i + 1, ' training accuracy: ', TrainingAcc: 2: 3);
  end;

  WriteLn('Traning completed. Now evaluating the model on the testing set...');

  DatasetTest := ReadCSV('../datasets/optdigits-test.csv');
  FeatsTest   := GetRange(DatasetTest, 0, 0, DatasetTest.Shape[0], 64) / 16;
  LabelsTest  := Squeeze(GetColumn(DatasetTest, 64, True));

  { Note that we do not need to wrap the test data in a variable, since we only
    need to evaluate the trained model. Thus, there is no need to create another
    computational graph. We can directly use FeatsTest as a TTensor, therefore we
    need to use the TTensor inside the model parameters, e.g., instead of using
    W1 directly, we shold use W1.Data }
  ypredTest := SoftMax(ReLU(FeatsTest.Dot(W1.Data) + b1.Data).Dot(W2.Data) +
    b2.Data, 1);

  TestingAcc := AccuracyScore(LabelEncoder.Decode(ypredTest), LabelsTest);
  WriteLn('testing accuracy = ', TestingAcc: 2: 2);

  { Displaying plot of training loss }
  ShowFigure(Losses, 'Training Loss Plot', ptLines);

  { Pick one sample from the test set. Let's try to visualize and predict the
    label }
  SampleIdx   := 850;
  ImageSample := GetRow(FeatsTest, SampleIdx, True);
  ypredTest   := SoftMax(ReLU(ImageSample.Dot(W1.Data) + b1.Data).Dot(W2.Data) +
    b2.Data, 1);

  { Reshape it first for display. }
  ImageSample.ReshapeInplace([8, 8]);

  { transform the probability into the discrete label }
  PredictedLabel := Round(LabelEncoder.Decode(ypredTest).Val[0]);
  ActualLabel    := Round(LabelsTest.GetAt(SampleIdx));

  { I don't know why the image is vertically flipped. So We should flip it back. }
  ShowFigure(VFlip(ImageSample), 'Predicted: ' + IntToStr(PredictedLabel) +
    '; Actual: ' + IntToStr(ActualLabel), ptImage);

  ReadLn;
end.

