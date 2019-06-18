unit DTMLUtils;

{ A unit containing machine-learning related classes anc functions }

{$mode delphi}

interface

uses
  Classes, SysUtils, fgl, Math, DTCore, DTUtils;

function Sigmoid(x: double): double; overload;
function Sigmoid(A: TDTMatrix): TDTMatrix; overload;
function SigmoidPrime(x: double): double; overload;
function SigmoidPrime(A: TDTMatrix): TDTMatrix; overload;
function Relu(x: double): double; overload;
function Relu(A: TDTMatrix): TDTMatrix; overload;
function ReluPrime(x: double): double; overload;
function ReluPrime(A: TDTMatrix): TDTMatrix; overload;
function AccuracyScore(yhat, y: TDTMatrix): double;

procedure TrainTestSplit(X, y: TDTMatrix; split: double; var XTrain: TDTMatrix;
  var XTest: TDTMatrix; var yTrain: TDTMatrix; var yTest: TDTMatrix;
  stratified: boolean);

type
  TBaseClassifier = class abstract(TObject)
  public
    function StartTraining(X, y: TDTMatrix): TBaseClassifier; virtual; abstract;
    function MakePrediction(X: TDTMatrix): TDTMatrix; virtual; abstract;
  end;

  { holding dataset statistic summaries, e.g., 'mean' and stdev }
  TSummaryMap = TFPGMap<string, TDTMatrix>;

  { holding dataset's class-wise statistic summaries, e.g., 'mean' and stdev }
  TClassWiseSummaryMap = TFPGMap<double, TSummaryMap>;

  TClassifierNaiveBayes = class(TBaseClassifier)
    ddof: integer;
    ClassWiseDatasetSummary: TClassWiseSummaryMap;
    uniqueLabels: TFloatVector;
  public
    constructor Create; overload;
    constructor Create(ddof: integer);
    function StartTraining(X, y: TDTMatrix): TClassifierNaiveBayes; override;
    function MakePrediction(X: TDTMatrix): TDTMatrix; override;
  private
    function CalculateGaussianPDF(x, _mean, _stdev: double): double;
  end;


implementation

constructor TClassifierNaiveBayes.Create;
begin
  { set the default degree of freedom to 1 for unbiased estimator }
  self.ddof := 1;
end;

function TClassifierNaiveBayes.CalculateGaussianPDF(x, _mean, _stdev: double): double;
var
  exponent: double;
begin
  exponent := exp(-(power(x - _mean, 2) / (2 * power(_stdev, 2))));
  Result := (1 / (sqrt(2 * PI) * _stdev)) * exponent;
end;

constructor TClassifierNaiveBayes.Create(ddof: integer);
begin
  self.ddof := ddof;
end;

function TClassifierNaiveBayes.StartTraining(X, y: TDTMatrix): TClassifierNaiveBayes;
var
  DatasetSummary, CSummary: TSummaryMap;
  ClassWiseData: TDTMatrix;
  j: integer;
  c: double;
begin
  { summarize overall dataset }
  DatasetSummary := TSummaryMap.Create;
  DatasetSummary.Add('mean', Mean(X, 0));
  DatasetSummary.Add('std', Std(X, 0, self.ddof));

  { summarize class-wise dataset }
  ClassWiseDatasetSummary := TClassWiseSummaryMap.Create;
  self.uniqueLabels := Getunique(y.val);
  { for each class }
  for c in self.uniqueLabels do
  begin
    ClassWiseData := CreateMatrix(0, X.Width);
    for j := 0 to X.Height - 1 do
    begin
      if y.val[j] = c then
        ClassWiseData := AppendRows(ClassWiseData, X.GetRow(j));
    end;
    CSummary := TSummaryMap.Create;
    CSummary.Add('mean', Mean(ClassWiseData, 0));
    CSummary.Add('std', Std(ClassWiseData, 0, self.ddof));
    ClassWiseDatasetSummary.Add(c, CSummary);
  end;

  Result := Self;
end;

function TClassifierNaiveBayes.MakePrediction(X: TDTMatrix): TDTMatrix;
var
  ProbabilityMap: TProbabilityMap;
  means, stdevs, instance, probs, preds: TDTMatrix;
  c, prob: double;
  i, row, cidx: integer;
begin
  preds := CreateMatrix(0, Length(self.uniqueLabels));
  for row := 0 to X.Height - 1 do
  begin
    instance := X.GetRow(row);
    probs := CreateMatrix(1, Length(self.uniqueLabels));
    cidx := 0;
    for c in self.uniqueLabels do
    begin
      prob := 1;
      means := TSummaryMap(ClassWiseDatasetSummary.KeyData[c]).KeyData['mean'];
      stdevs := TSummaryMap(ClassWiseDatasetSummary.KeyData[c]).KeyData['std'];
      for i := 0 to X.Width - 1 do
      begin
        { calculate probability of each class of each instance }
        prob := prob * CalculateGaussianPDF(instance.val[i], means.val[i],
          stdevs.val[i]);
      end;
      probs.val[cidx] := prob;
      Inc(cidx);
    end;
    preds := AppendRows(preds, probs);
  end;

  { the actual class label for each instance }
  Result := IndexMax(preds, 1);
end;

{ @abstract(Sigmoid activation function) }
function Sigmoid(x: double): double;
begin
  Result := 1 / (1 + exp(-x));
end;

function Sigmoid(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@Sigmoid, A);
end;

function SigmoidPrime(x: double): double;
begin
  Result := Sigmoid(x) * (1 - Sigmoid(x));
end;

function SigmoidPrime(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@SigmoidPrime, A);
end;

function Relu(x: double): double;
begin
  Result := x * integer(x >= 0);
end;

function Relu(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@Relu, A);
end;

function ReluPrime(x: double): double;
begin
  Result := 1 * integer(x >= 0);
end;

function ReluPrime(A: TDTMatrix): TDTMatrix;
begin
  Result := Apply(@ReluPrime, A);
end;

function AccuracyScore(yhat, y: TDTMatrix): double;
var
  i, tot: integer;
begin
  tot := 0;
  for i := 0 to Length(y.val) - 1 do
    if y.val[i] = yhat.val[i] then
      tot := tot + 1;
  Result := tot / Length(y.val);
end;

procedure TrainTestSplit(X, y: TDTMatrix; split: double; var XTrain: TDTMatrix;
  var XTest: TDTMatrix; var yTrain: TDTMatrix; var yTest: TDTMatrix;
  stratified: boolean);
var
  TrainSize, index: integer;
  XCopy, yCopy: TDTMatrix;
begin
  XTrain := CreateMatrix(0, X.Width);
  yTrain := CreateMatrix(0, y.Width);
  TrainSize := round(split * X.Height);
  XCopy := CopyMatrix(X);
  yCopy := CopyMatrix(y);
  while Xtrain.Height < TrainSize do
  begin
    index := Random(XCopy.Height);
    Xtrain := AppendRows(Xtrain, PopRow(XCopy, index));
    ytrain := AppendRows(ytrain, PopRow(yCopy, index));
  end;
  XTest := XCopy;
  yTest := yCopy;
end;

end.
