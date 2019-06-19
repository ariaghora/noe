unit Unit1;

{$mode objfpc}{$H+}

interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls,
  DTCore, DTMLUtils, DTPreprocessing;

type

  { TForm1 }

  TForm1 = class(TForm)
    BtnLogisticRegression: TButton;
    ButtonNaiveBayes: TButton;
    Button3: TButton;
    LabelFilename: TLabel;
    MemoOutput: TMemo;
    procedure BtnLogisticRegressionClick(Sender: TObject);
    procedure ButtonNaiveBayesClick(Sender: TObject);
    procedure FormClose(Sender: TObject; var CloseAction: TCloseAction);
    procedure FormCreate(Sender: TObject);
  private

  public

  end;

var
  Form1: TForm1;
  filename: string;
  Dataset, X, y, XTrain, Xtest, yTrain, yTest: TDTMatrix;
  Predictions: TDTMatrix;
  TrainAccuracy, TestAccuracy: double;
  NaiveBayes: TClassifierNaiveBayes;
  LogisticRegression: TClassifierLogisticRegression;
  Scaler: TMinMaxScaler;

implementation

{$R *.lfm}

{ TForm1 }

procedure TForm1.FormCreate(Sender: TObject);
begin
  filename := 'iris_dataset.csv';
  LabelFilename.Caption := 'Dataset: ' + filename;

  { Intialize darkteal }
  DarkTealInit;

  { Load dataset }
  Dataset := TDTMatrixFromCSV(filename);
  X := Dataset.GetRange(0, 0, Dataset.Height, Dataset.Width - 1); // features
  y := Dataset.GetColumn(Dataset.Width - 1); // label

  { Split the dataset into training set and testing set }
  TrainTestSplit(X, y, 0.7, XTrain, Xtest, yTrain, yTest, False);

  { Scale the dataset }
  Scaler := TMinMaxScaler.Create;
  Scaler.Fit(XTrain);
  XTrain := Scaler.Transform(XTrain);
  XTest := Scaler.Transform(Xtest);
end;

procedure TForm1.BtnLogisticRegressionClick(Sender: TObject);
begin
  Randomize;
  LogisticRegression := TClassifierLogisticRegression.Create;

  { You may also change the default training iteration and learning rate }
  LogisticRegression.NIter := 1000;
  LogisticRegression.LearningRate := 0.01;

  MemoOutput.Lines.Clear;
  MemoOutput.Lines.Add('Training Logistic Regression classifier...');

  LogisticRegression.StartTraining(XTrain, yTrain);
  Predictions := LogisticRegression.MakePrediction(XTrain);
  TrainAccuracy := AccuracyScore(Predictions, yTrain);
  Predictions := LogisticRegression.MakePrediction(Xtest);
  TestAccuracy := AccuracyScore(Predictions, yTest);

  MemoOutput.Lines.Add('Finish training');
  MemoOutput.Lines.Add('Training accuracy : ' + FormatFloat('#.##',
    TrainAccuracy * 100) + '%');
  MemoOutput.Lines.Add('Testing accuracy  : ' + FormatFloat('#.##',
    TestAccuracy * 100) + '%');
end;

procedure TForm1.ButtonNaiveBayesClick(Sender: TObject);
begin
  NaiveBayes := TClassifierNaiveBayes.Create;

  MemoOutput.Lines.Clear;
  MemoOutput.Lines.Add('Training Naive Bayes classifier...');

  NaiveBayes.StartTraining(XTrain, yTrain);
  Predictions := NaiveBayes.MakePrediction(XTrain);
  TrainAccuracy := AccuracyScore(Predictions, yTrain);
  Predictions := NaiveBayes.MakePrediction(Xtest);
  TestAccuracy := AccuracyScore(Predictions, yTest);

  MemoOutput.Lines.Add('Finish training');
  MemoOutput.Lines.Add('Training accuracy : ' + FormatFloat('#.##',
    TrainAccuracy * 100) + '%');
  MemoOutput.Lines.Add('Testing accuracy  : ' + FormatFloat('#.##',
    TestAccuracy * 100) + '%');
end;

procedure TForm1.FormClose(Sender: TObject; var CloseAction: TCloseAction);
begin
  { Cleanup darkteal }
  DarkTealRelease;
end;

end.




