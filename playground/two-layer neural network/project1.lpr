{ This demo program shows the low-level usage of darkteal to build a two-layer
  neural network for iris dataset classification. }

program project1;

uses
  DTCore,
  DTPreprocessing,
  DTMLUtils;

var
  { model parameters }
  W0, W1: TDTMatrix;
  Layer0: TDTMatrix;
  Layer1, Layer1Error, Layer1Delta: TDTMatrix;
  Layer2, Layer2Error, Layer2Delta: TDTMatrix;

  dataset, X, y, yBin, XTrain, yTrain, XTest, yTest, prediction: TDTMatrix;
  encoder: TOneHotEncoder;
  LearningRate, Accuracy: double;
  iter, MAX_ITER, NHiddenNeuron: integer;

begin
  RandSeed := 123;

  { Load the dataset }
  dataset := TDTMatrixFromCSV('../../datasets/fisher_iris.csv');
  X := dataset.GetRange(0, 0, dataset.Height, dataset.Width - 1);
  y := dataset.GetColumn(dataset.Width - 1);

  TrainTestSplit(X, y, 0.7, XTrain, XTest, yTrain, yTest, False);

  { one-hot encode the training labels }
  encoder := TOneHotEncoder.Create.Fit(yTrain);
  yBin := encoder.Transform(yTrain);

  { hyper-parameter setting }
  LearningRate := 0.01;
  NHiddenNeuron := 5;
  MAX_ITER := 1000;

  { initialize model parameters }
  W0 := 2 * CreateMatrix(XTrain.Width, NHiddenNeuron) - 1;
  W1 := 2 * CreateMatrix(NHiddenNeuron, yBin.Width) - 1;

  Layer0 := XTrain;

  { start gradient descent}
  for iter := 1 to MAX_ITER do
  begin
    { feed-forward }
    Layer1 := Sigmoid(Layer0.Dot(W0));
    Layer2 := Sigmoid(Layer1.Dot(W1));

    { backpropagation }
    Layer2Error := yBin - Layer2;
    Layer2Delta := Layer2Error * SigmoidPrime(Layer2);
    Layer1Error := Layer2Delta.Dot(W1.T);
    Layer1Delta := Layer1Error * SigmoidPrime(Layer1);

    { weight update }
    W1 := W1 + Layer1.T.Dot(Layer2Delta) * LearningRate;
    W0 := W0 + Layer0.T.Dot(Layer1Delta) * LearningRate;

    if iter mod 10 = 0 then
      WriteLn('Error at iteration ', iter, ' = ',
        Mean(Power(Layer2Error, 2)));
  end;
  WriteLn('Training finished!');

  { make predictions }
  prediction := Sigmoid(Sigmoid(XTest.Dot(W0)).Dot(W1));
  Accuracy := AccuracyScore(IndexMax(prediction, 1), yTest);

  { print test accuracy }
  WriteLn('Testing accuracy: ', Accuracy * 100: 2: 2, '%');

  ReadLn;
end.
